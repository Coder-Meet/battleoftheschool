from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import research_corpus as corpus
from candidate_patches import EXTRA_FEATURE_NAMES
from detector import Branch, Detection, DetectorConfig
from evaluate import evaluate_case
from learning import CandidateModel, FEATURE_NAMES, features
from synthetic_reviews import label_candidates


def branch(identifier: str, x: float) -> Branch:
    return Branch(
        identifier, (x, 0, 0), (x + 5, 0, 0), 2, (1, 0, 0),
        [(x, 0, 0), (x + 10, 0, 0)], 0.8, 0.5,
        features={
            "path_hu_relative": 1, "bone_distance_mm": 20, "parent_angle_degrees": 90,
            "arc_position": 0.5, "native_spacing_mm": 1, "connector_gap": 0, "candidate_volume_mm3": 100,
        },
    )


def detection(branches: list[Branch]) -> Detection:
    return Detection(branches, {"total_s": 0}, {}, len(branches), {}, [], DetectorConfig())


@pytest.fixture
def planned(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    corpus.make_plan(root)
    return root


def test_preregistration_has_210_unique_cases_and_disjoint_seed_groups(planned: Path) -> None:
    plan = corpus.load_plan(planned)
    assert len(plan["cases"]) == 210
    assert len({r["case_id"] for r in plan["cases"]}) == 210
    assert not set(plan["seeds"]) & corpus.FORBIDDEN_SEEDS
    groups = []
    for part, size in zip(corpus.PARTITIONS, (10, 3, 2)):
        selected = [r for r in plan["cases"] if r["partition"] == part]
        assert len(selected) == size * 14
        assert set(r["family"] for r in selected) == set(corpus.FAMILIES)
        groups.append(set(r["group_id"] for r in selected))
        assert len(groups[-1]) == size
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert plan["seeds"] == corpus.new_seeds()
    assert not (planned / "exports").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "exposed", "negative", "bool", "short"])
def test_invalid_seed_groups_are_rejected(mutation: str) -> None:
    seeds = corpus.new_seeds()
    if mutation == "duplicate":
        seeds[-1] = seeds[0]
    elif mutation == "exposed":
        seeds[-1] = 4001
    elif mutation == "negative":
        seeds[-1] = -1
    elif mutation == "bool":
        seeds[-1] = True
    else:
        seeds.pop()
    with pytest.raises(ValueError, match="fresh"):
        corpus.case_records(seeds)


def test_plan_rejects_group_leakage_source_and_preprocessing_drift(planned: Path) -> None:
    plan = corpus.load_plan(planned)
    for key, value in (
        ("source_sha256", {}), ("preprocessing", {}), ("versions", {}),
        ("python", "0.0.0"), ("strict_config", {}),
    ):
        changed = deepcopy(plan)
        changed[key] = value
        with pytest.raises(ValueError, match="drift"):
            corpus.validate_plan(changed)
    changed = deepcopy(plan)
    changed["cases"][0]["partition"] = "test"
    with pytest.raises(ValueError, match="drift"):
        corpus.validate_plan(changed)
    with pytest.raises(FileExistsError):
        corpus.make_plan(planned)
    with pytest.raises(FileExistsError):
        corpus.save(planned / "plan.json", {})


def test_content_and_split_integrity_fail_closed(planned: Path) -> None:
    split_path = planned / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        corpus.load_plan(planned)
    split_path.with_name("split.json.sha256").unlink()
    corpus.seal(split_path)
    with pytest.raises(ValueError, match="Split drift"):
        corpus.load_plan(planned)


def test_optional_training_dependencies_are_locked_without_requiring_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_version = corpus.importlib.metadata.version
    optional = {"scikit-learn", "joblib", "threadpoolctl"}

    def absent(name: str) -> str:
        if name in optional:
            raise corpus.importlib.metadata.PackageNotFoundError(name)
        return installed_version(name)

    monkeypatch.setattr(corpus.importlib.metadata, "version", absent)
    root = tmp_path / "without-training"
    corpus.make_plan(root)
    assert all(corpus.load_plan(root)["versions"][name] == "not-installed" for name in optional)

    def changed(name: str) -> str:
        return "1.7.2" if name == "scikit-learn" else absent(name)

    monkeypatch.setattr(corpus.importlib.metadata, "version", changed)
    with pytest.raises(ValueError, match="dependency integrity drift"):
        corpus.load_plan(root)


def test_case_resume_rejects_changed_and_incomplete_transactions(tmp_path: Path) -> None:
    directory = tmp_path / "case"
    directory.mkdir()
    path = directory / "reference.json"
    corpus.write_json(path, {"daughters": []})
    with pytest.raises(FileNotFoundError):
        corpus.complete_directory(directory, "input.json")
    corpus.save(directory / "input.json", {"files": {"reference.json": corpus.digest(path)}})
    corpus.complete_directory(directory, "input.json")
    path.write_text('{"daughters": [1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        corpus.complete_directory(directory, "input.json")


def test_test_export_requires_frozen_models_before_reading_any_case(planned: Path) -> None:
    plan = corpus.load_plan(planned)
    test_case = next(r for r in plan["cases"] if r["partition"] == "test")
    with pytest.raises(FileNotFoundError, match="freeze"):
        corpus.export_one((str(planned), test_case, 1, False))
    assert not (planned / "exports").exists()


def test_freeze_rejects_model_and_training_input_drift(planned: Path) -> None:
    directory = planned / "models"
    directory.mkdir()
    model = directory / "tree.json"
    reviews = planned / "reviews-development.json"
    corpus.write_json(model, {"threshold": 0.5})
    corpus.write_json(reviews, {"records": []})
    corpus.save(directory / "freeze.json", {
        "plan_sha256": corpus.digest(planned / "plan.json"),
        "files": {"tree.json": corpus.digest(model)},
        "inputs": {"reviews-development.json": corpus.digest(reviews)},
    })
    corpus.verify_freeze(planned)
    original = model.read_bytes()
    model.write_text('{"threshold": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="model integrity"):
        corpus.verify_freeze(planned)
    model.write_bytes(original)
    reviews.write_text('{"records": [1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="input drift"):
        corpus.verify_freeze(planned)


def test_filtering_keeps_real_unproposed_misses_and_never_relabels_ambiguous_proposals() -> None:
    pool = detection([branch("good", 0), branch("ambiguous", 4), branch("negative", 80)])
    reference = detection([branch("ref1", 0), branch("missed", 30)]).prediction("case")
    rows, report = label_candidates(pool, reference)
    assert {r["instance_id"] for r in rows} == {"good", "negative"}
    assert report["excluded_ambiguous_candidate_ids"] == ["ambiguous"]
    model = CandidateModel(
        [0.0] * 13, [1.0] * 13, [0.0] * 13, 0, 1,
        {"train": ["a"], "validation": ["b"], "test": ["c"]},
    )
    prediction = pool.prediction("case")
    all_rows = [{"instance_id": b.instance_id, "features": features(b)} for b in pool.branches]
    filtered = corpus.filter_prediction(prediction, all_rows, model)
    metrics = evaluate_case(filtered, reference, 3)
    assert metrics["false_negatives"] == 2
    assert metrics["unmatched_reference_ids"] == ["ref1", "missed"]
    assert len(prediction["daughters"]) == 3
    comparison = corpus.comparisons([report], [metrics], {"case": "synthetic:1"})
    assert comparison["induced_misses"] == 1
    assert comparison["lost_reference_ids"] == {"case": ["ref1"]}
    assert comparison["fp_reduction"] == 2
    with pytest.raises(ValueError, match="identity"):
        corpus.filter_prediction(prediction, all_rows[::-1], model)
    with pytest.raises(ValueError, match="every case"):
        corpus.comparisons([report], [], {"case": "synthetic:1"})


def test_empty_cases_remain_in_evaluation_with_undefined_negative_recall() -> None:
    empty = detection([]).prediction("case")
    metrics = evaluate_case(empty, empty, 3)
    compared = corpus.comparisons([metrics], [metrics], {"case": "synthetic:1"})
    assert compared["paired_bootstrap"]["cases"] == ["case"]
    assert compared["fp_reduction_fraction"] is None
    assert metrics["recall"] is None


def test_partition_patches_are_numeric_ordered_and_exclude_only_ambiguous_training_rows(tmp_path: Path) -> None:
    root = tmp_path
    corpus.write_json(root / "manifest.json", {"source": "analytic_synthetic_geometry"})
    records = []
    for i, part in enumerate(("train", "validation", "test")):
        for empty in (False, True):
            case_id = f"{part}-{empty}"
            records.append({"case_id": case_id, "partition": part})
            directory = root / "exports" / case_id
            directory.mkdir(parents=True)
            rows = [] if empty else [{
                "instance_id": "real-negative", "local_patch_index": 1, "features": [1.0] * 13,
                "extra_features": dict(zip(EXTRA_FEATURE_NAMES, (1.0, 2.0, 3.0))),
                "fingerprint": f"candidate-{i}", "case_id": case_id,
            }]
            corpus.write_json(directory / "candidates.json", {
                "case_id": case_id, "all_candidates": [] if empty else [{}, {}], "records": rows,
                "pool_metrics": {"unmatched_reference_ids": ["not-a-candidate"], "false_negatives": 1},
            })
            patches = np.empty((0 if empty else 2, 12, 32, 32), dtype=np.float32)
            if not empty:
                patches[0].fill(-99)
                patches[1].fill(i + 10)
            with (directory / "patches.npz").open("xb") as stream:
                np.savez_compressed(stream, patches=patches)
            corpus.save(directory / "receipt.json", {
                "files": {p.name: corpus.digest(p) for p in directory.iterdir()},
            })
    plan = {"cases": records, "maximum_patch_partition_bytes": 1000000, "preprocessing": {}, "source_sha256": {}}
    for i, part in enumerate(corpus.PARTITIONS):
        corpus.assemble_partition(root, part, plan)
        with np.load(root / f"patches-{part}.npz", allow_pickle=False) as archive:
            assert archive.files == ["patches"]
            np.testing.assert_array_equal(archive["patches"], np.full((1, 12, 32, 32), i + 10, dtype=np.float32))
        metadata = corpus.read_json(root / f"patches-{part}.json")
        assert len(metadata["cases"]) == 2
        assert metadata["records"][0]["patch_index"] == 0
        assert metadata["records"][0]["features"] == [1.0] * len(FEATURE_NAMES)
        assert metadata["records"][0]["fingerprint"] == f"candidate-{i}"
        assert all(case["pool_metrics"]["unmatched_reference_ids"] == ["not-a-candidate"] for case in metadata["cases"])


@pytest.mark.parametrize("array,count", [
    (np.zeros((1, 12, 32, 32), dtype=np.float64), 1),
    (np.zeros((1, 3, 32, 32), dtype=np.float32), 1),
    (np.full((1, 12, 32, 32), np.nan, dtype=np.float32), 1),
    (np.zeros((1, 12, 32, 32), dtype=np.float32), 0),
])
def test_corrupt_patch_cache_rejected(array: np.ndarray, count: int) -> None:
    with pytest.raises(ValueError, match="patch cache"):
        corpus.validate_patches(array, count)


def test_cnn_gate_requires_real_fp_gain_and_no_lost_branches() -> None:
    run: dict = {
        "versus_pool": {"induced_misses": 0, "fp_reduction": 2, "fp_reduction_fraction": 0.2},
        "versus_strict": {"induced_misses": 0},
        "summary": {"false_positives": 8, "false_negatives": 2},
    }
    logistic = {"summary": {"false_positives": 9, "false_negatives": 2}}
    assert corpus.gate(run, logistic)
    assert not corpus.gate(run, run)
    for comparison in ("versus_pool", "versus_strict"):
        changed = deepcopy(run)
        changed[comparison]["induced_misses"] = 1
        assert not corpus.gate(changed, logistic)
    changed = deepcopy(run)
    changed["versus_pool"]["fp_reduction_fraction"] = 0.01
    assert not corpus.gate(changed, logistic)
