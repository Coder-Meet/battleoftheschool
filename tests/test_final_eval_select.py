import copy
from pathlib import Path

import pytest

import final_eval_select as select
from final_evaluation import CASES, score_variant, summarize, write_json


def score_row(case: str, true_positive: int) -> dict:
    return {
        "case_id": case,
        "true_positives": true_positive,
        "false_positives": 0,
        "false_negatives": 1 - true_positive,
        "daughter_counts": {
            "predicted": true_positive,
            "reference": 1,
            "signed_error": true_positive - 1,
            "absolute_error": 1 - true_positive,
        },
        "precision": float(true_positive) if true_positive else None,
        "recall": float(true_positive),
        "f1": float(true_positive),
        "matches": [],
        "unmatched_prediction_ids": [],
        "unmatched_reference_ids": [],
    }


def normalized_variant(
    name: str,
    successes: list[int],
    *,
    eligible: bool = True,
    dependencies: int = 0,
    components: int = 1,
    runtime: float = 1.0,
) -> dict:
    rows = [score_row(case, value) for case, value in zip(CASES, successes)]
    scores = {
        tolerance: {"summary": summarize(copy.deepcopy(rows)), "cases": copy.deepcopy(rows)}
        for tolerance in select.TOLERANCE_KEYS
    }
    predictions = {
        case: {"case_id": case, "parent": {"instance_id": "aorta"}, "daughters": []}
        for case in CASES
    }
    return {
        "family": "fixture",
        "name": name,
        "global_name": f"fixture:{name}",
        "configuration": {"name": name},
        "clean_frozen_selection_eligible": eligible,
        "report_eligible": eligible,
        "origin_policy_certified": True,
        "deployment_eligible": False,
        "ineligible_reason": "" if eligible else "fixture exclusion",
        "scores": scores,
        "runtime": {case: {"runtime_s": runtime, "peak_rss_mb": 1.0} for case in CASES},
        "complexity": {
            "effective_model_dependencies": dependencies,
            "algorithm_components": components,
            "dependency_ids": [],
        },
        "predictions": predictions,
        "prediction_provenance": {
            "directory": f"fixture/{name}",
            "byte_sha256": dict.fromkeys(CASES, "a" * 64),
            "semantic_sha256": dict.fromkeys(CASES, "b" * 64),
            "set_semantic_sha256": name,
        },
    }


def test_held_out_scores_runtime_and_prediction_cannot_change_fold_choice():
    first = normalized_variant("first", [0, 1, 1, 1, 1])
    second = normalized_variant("second", [1, 0, 0, 0, 0])
    before, _ = select.select_loco([first, second])
    assert before["folds"][0]["selected_global_name"] == "fixture:first"

    poisoned = copy.deepcopy(first)
    held_out = CASES[0]
    poisoned["scores"]["3"]["cases"][0] = score_row(held_out, 100)
    poisoned["runtime"][held_out]["runtime_s"] = 1_000_000
    poisoned["predictions"][held_out]["poison"] = "not consulted by ranking"
    after, predictions = select.select_loco([poisoned, second])
    assert after["folds"][0]["selected_global_name"] == "fixture:first"
    assert predictions[held_out]["poison"] == "not consulted by ranking"


def test_ineligible_perfect_row_never_selected_and_order_is_invariant():
    first = normalized_variant("first", [0, 1, 1, 1, 1])
    second = normalized_variant("second", [1, 0, 0, 0, 0])
    excluded = normalized_variant("perfect", [1, 1, 1, 1, 1], eligible=False)
    forward, _ = select.select_loco([first, second, excluded])
    reverse, _ = select.select_loco([excluded, second, first])
    assert forward == reverse
    assert all(fold["selected_global_name"] != "fixture:perfect" for fold in forward["folds"])


def test_rank_ties_apply_complexity_runtime_then_lexical_name():
    scores = [1, 1, 1, 1, 1]
    expensive = normalized_variant("a-expensive", scores, dependencies=1, runtime=0.01)
    simple = normalized_variant("z-simple", scores, dependencies=0, components=2, runtime=100)
    assert select.rank_key(simple, CASES) < select.rank_key(expensive, CASES)

    fewer_components = normalized_variant("z-components", scores, components=1, runtime=100)
    assert select.rank_key(fewer_components, CASES) < select.rank_key(simple, CASES)

    faster = normalized_variant("z-fast", scores, components=1, runtime=1)
    assert select.rank_key(faster, CASES) < select.rank_key(fewer_components, CASES)

    lexical_a = normalized_variant("a", scores, components=1, runtime=1)
    lexical_b = normalized_variant("b", scores, components=1, runtime=1)
    assert select.rank_key(lexical_a, CASES) < select.rank_key(lexical_b, CASES)


def test_paired_bootstrap_is_deterministic_paired_and_zero_for_identical_scores():
    variant = normalized_variant("same", [0, 1, 1, 0, 1])
    first = select.paired_bootstrap(variant["scores"], variant["scores"], replicates=200)
    second = select.paired_bootstrap(variant["scores"], variant["scores"], replicates=200)
    assert first == second
    assert all(
        row["paired_f1_delta_95_percentile"] == [0, 0]
        and row["paired_count_mae_delta_95_percentile"] == [0, 0]
        for row in first["tolerances"].values()
    )
    reversed_scores = copy.deepcopy(variant["scores"])
    reversed_scores["3"]["cases"].reverse()
    with pytest.raises(ValueError, match="identical ordered whole cases"):
        select.paired_bootstrap(variant["scores"], reversed_scores, replicates=10)


def make_fixture_reports(tmp_path: Path) -> tuple[dict[str, dict], dict[str, int], dict[str, int]]:
    predictions = {
        case: {"case_id": case, "parent": {"instance_id": "aorta"}, "daughters": []}
        for case in CASES
    }
    scores = score_variant(predictions)
    reports = {}
    expected = {}
    eligible_counts = {}
    for family in select.REPORT_PATHS:
        name = f"{family}-fixture"
        directory = tmp_path / "predictions" / family / name
        for case, prediction in predictions.items():
            write_json(directory / f"{case}.json", prediction)
        eligible = family == "deterministic"
        if family == "deterministic":
            config = {"origin_size_eligibility_disabled": False, "model_hashes": {}}
        elif family == "tabular":
            config = {"profile": "strict", "model": None}
        elif family == "cnn":
            config = {"family": "current", "proposals": "strict", "unfiltered": True}
        else:
            config = {"detector": {"minimum_origin_diameter_mm": 2}}
        report = {
            "schema_version": 1,
            "family": family,
            "variants": [{
                "name": name,
                "prediction_dir": directory.relative_to(tmp_path).as_posix(),
                "configuration": config,
                "eligible_for_selection": eligible,
                "ineligible_reason": "" if eligible else "fixture exclusion",
                "scores": copy.deepcopy(scores),
                "runtime": {
                    case: {"runtime_s": 1.0, "peak_rss_mb": 2.0} for case in CASES
                },
            }],
            "failures": [],
        }
        if family == "deterministic":
            report["worker_failures"] = {"failures": []}
        reports[family] = report
        expected[family] = 1
        eligible_counts[family] = int(eligible)
    return reports, expected, eligible_counts


def validate_fixture(tmp_path: Path, reports: dict, expected: dict, eligible: dict) -> list[dict]:
    variants, _ = select.validate_inventory(
        reports,
        tmp_path,
        expected_variants=expected,
        expected_eligible=eligible,
        require_repository_pins=False,
        fixture_mode=True,
    )
    return variants


def test_inventory_replay_accepts_empty_prediction_cases_and_all_families(tmp_path: Path):
    reports, expected, eligible = make_fixture_reports(tmp_path)
    variants = validate_fixture(tmp_path, reports, expected, eligible)
    assert len(variants) == 5
    assert all(set(row["predictions"]) == set(CASES) for row in variants)
    deterministic = next(row for row in variants if row["family"] == "deterministic")
    assert deterministic["scores"]["3"]["summary"]["false_negatives"] == 19


def test_inventory_rejects_missing_family_variant_and_duplicate_name(tmp_path: Path):
    reports, expected, eligible = make_fixture_reports(tmp_path)
    missing_family = copy.deepcopy(reports)
    del missing_family["audit"]
    with pytest.raises(ValueError, match="Every declared family"):
        validate_fixture(tmp_path, missing_family, expected, eligible)

    missing_variant = copy.deepcopy(reports)
    missing_variant["cnn"]["variants"] = []
    with pytest.raises(ValueError, match="exactly 1 variants"):
        validate_fixture(tmp_path, missing_variant, expected, eligible)

    duplicate = copy.deepcopy(reports)
    duplicate["audit"]["variants"][0]["name"] = duplicate["topology"]["variants"][0]["name"]
    with pytest.raises(ValueError, match="globally unique"):
        validate_fixture(tmp_path, duplicate, expected, eligible)


def test_inventory_rejects_case_tolerance_runtime_and_cached_score_drift(tmp_path: Path):
    reports, expected, eligible = make_fixture_reports(tmp_path)
    missing_case = copy.deepcopy(reports)
    missing_case["tabular"]["variants"][0]["scores"]["3"]["cases"].pop()
    with pytest.raises(ValueError, match="incomplete 3-mm"):
        validate_fixture(tmp_path, missing_case, expected, eligible)

    missing_tolerance = copy.deepcopy(reports)
    del missing_tolerance["tabular"]["variants"][0]["scores"]["5"]
    with pytest.raises(ValueError, match="exactly 2/3/5-mm"):
        validate_fixture(tmp_path, missing_tolerance, expected, eligible)

    nonfinite_runtime = copy.deepcopy(reports)
    nonfinite_runtime["cnn"]["variants"][0]["runtime"][CASES[0]]["runtime_s"] = float("nan")
    with pytest.raises(ValueError, match="non-finite runtime"):
        validate_fixture(tmp_path, nonfinite_runtime, expected, eligible)

    drifted_score = copy.deepcopy(reports)
    drifted_score["audit"]["variants"][0]["scores"]["3"]["summary"]["f1"] = 0.5
    with pytest.raises(ValueError, match="Cached score drift"):
        validate_fixture(tmp_path, drifted_score, expected, eligible)


def test_inventory_rejects_path_escape_case_id_hash_and_nonfinite_score(tmp_path: Path):
    reports, expected, eligible = make_fixture_reports(tmp_path)
    traversal = copy.deepcopy(reports)
    traversal["deterministic"]["variants"][0]["prediction_dir"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        validate_fixture(tmp_path, traversal, expected, eligible)

    wrong_case = copy.deepcopy(reports)
    path = tmp_path / wrong_case["audit"]["variants"][0]["prediction_dir"] / f"{CASES[0]}.json"
    prediction = select.strict_read_json(path)
    prediction["case_id"] = "wrong"
    write_json(path, prediction)
    with pytest.raises(ValueError, match="case ID mismatch"):
        validate_fixture(tmp_path, wrong_case, expected, eligible)

    reports, expected, eligible = make_fixture_reports(tmp_path)
    hash_drift = copy.deepcopy(reports)
    hash_drift["deterministic"]["variants"][0]["prediction_hashes"] = dict.fromkeys(CASES, "0" * 64)
    with pytest.raises(ValueError, match="recorded prediction hashes"):
        validate_fixture(tmp_path, hash_drift, expected, eligible)

    nonfinite_score = copy.deepcopy(reports)
    nonfinite_score["topology"]["variants"][0]["scores"]["3"]["summary"]["f1"] = float("inf")
    with pytest.raises(ValueError, match="Non-finite"):
        validate_fixture(tmp_path, nonfinite_score, expected, eligible)


def test_repository_pin_check_fails_on_scorer_drift(tmp_path: Path):
    (tmp_path / "final_evaluation.py").write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="scorer hash drifted"):
        select.validate_repository_pins(tmp_path)


def test_real_inventory_replays_all_280_variants_and_detects_strict_aliases():
    reports, _ = select.load_reports()
    variants, inventory = select.validate_inventory(reports)
    assert len(variants) == 280
    assert sum(row["clean_frozen_selection_eligible"] for row in variants) == 180
    assert inventory["families"] == {
        "deterministic": {"variants": 52, "report_eligible": 12},
        "tabular": {"variants": 106, "report_eligible": 84},
        "cnn": {"variants": 112, "report_eligible": 84},
        "topology": {"variants": 9, "report_eligible": 0},
        "audit": {"variants": 1, "report_eligible": 0},
    }
    strict = select.find_variant(variants, select.STRICT_ID)
    alias = next(
        row for row in select.semantic_aliases(variants)
        if row["semantic_prediction_set_sha256"]
        == strict["prediction_provenance"]["set_semantic_sha256"]
    )
    assert {
        "deterministic:deterministic-strict",
        "tabular:tabular-unfiltered-strict",
        "cnn:cnn-current-strict-unfiltered",
        "audit:audit_strict_scoring_control",
    }.issubset(alias["global_names"])
    assert alias["clean_eligibility_conflict"]


def test_pinned_file_identity_accepts_payload_and_canonical_lfs_pointer(tmp_path: Path):
    path = tmp_path / "input.nii"
    payload = b"resolved evaluation payload"
    expected_sha256 = select.hashlib.sha256(payload).hexdigest()
    path.write_bytes(payload)
    assert select.validate_pinned_file_identity(path, expected_sha256, len(payload)) == {
        "sha256": expected_sha256,
        "size": len(payload),
        "representation": "resolved_payload",
        "payload_resolved": True,
    }

    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{expected_sha256}\n"
        f"size {len(payload)}\n"
    ).encode("ascii")
    path.write_bytes(pointer)
    assert select.validate_pinned_file_identity(path, expected_sha256, len(payload)) == {
        "sha256": expected_sha256,
        "size": len(payload),
        "representation": "canonical_git_lfs_pointer",
        "payload_resolved": False,
    }


def test_pinned_file_identity_rejects_wrong_or_malformed_lfs_pointers(tmp_path: Path):
    path = tmp_path / "input.nii"
    expected_sha256 = "1" * 64
    expected_size = 123
    invalid = (
        (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{'0' * 64}\n"
            f"size {expected_size}\n"
        ),
        (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{expected_sha256}\n"
            f"size {expected_size + 1}\n"
        ),
        (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{expected_sha256}\n"
            f"size {expected_size}\n"
            "extra metadata\n"
        ),
        (
            "version https://git-lfs.github.com/spec/v1\r\n"
            f"oid sha256:{expected_sha256}\r\n"
            f"size {expected_size}\r\n"
        ),
    )
    for pointer in invalid:
        path.write_bytes(pointer.encode("ascii"))
        with pytest.raises(ValueError, match="Git LFS pointer identity"):
            select.validate_pinned_file_identity(path, expected_sha256, expected_size)


def test_count_preserving_tabular_eligibility_swap_is_rejected():
    reports, _ = select.load_reports()
    mutated = dict(reports)
    tabular = dict(reports["tabular"])
    rows = list(tabular["variants"])
    dirty_name = "tabular-original-logistic-strict-t0.14999999999999999"
    dirty_index = next(index for index, row in enumerate(rows) if row["name"] == dirty_name)
    clean_index = next(
        index for index, row in enumerate(rows)
        if row["eligible_for_selection"] and row["name"] != dirty_name
    )
    assert rows[dirty_index]["eligible_for_selection"] is False
    rows[dirty_index] = {**rows[dirty_index], "eligible_for_selection": True}
    rows[clean_index] = {**rows[clean_index], "eligible_for_selection": False}
    assert sum(row["eligible_for_selection"] for row in rows) == 84
    tabular["variants"] = rows
    mutated["tabular"] = tabular

    with pytest.raises(ValueError, match="eligibility contradicts reconstructed family evidence"):
        select.validate_inventory(mutated)


def strict_contract_fixture() -> tuple[dict, dict, dict]:
    source_hashes = {
        "detector.py": select.PINNED_DETECTOR_SHA256,
        "final_evaluation.py": select.PINNED_SCORER_SHA256,
    }
    strict = {
        "configuration": {
            "engine": "detect",
            "detector_config": select.asdict(select.DetectorConfig()),
            "paper_method": "",
            "model_hashes": {},
            "strict_union_config": None,
            "category": "frozen_existing_algorithm",
            "trained_on_reference_cases": False,
            "post_reference_new_algorithm": False,
            "origin_size_eligibility_disabled": False,
            "candidate_threshold": None,
            "operation": "final",
            "source_hashes": dict(source_hashes),
            "threads": 4,
        }
    }
    inventory = {
        "deterministic_matrix_source_hashes": dict(source_hashes),
        "strict_frozen_pins_validated": True,
        "deterministic_replay": {"verdict": "PASS"},
    }
    topology_gate = {
        "criteria": {"exact_checkpoint_set": "11 variants x 24 frozen-derived cases = 264"},
        "verdict": "PASS: frozen checkpoint replay fixture",
    }
    return strict, inventory, topology_gate


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("detector_config", "minimum_origin_diameter_mm"), 0.0),
        (("source_hashes", "detector.py"), "0" * 64),
        (("trained_on_reference_cases",), True),
        (("post_reference_new_algorithm",), True),
        (("candidate_threshold",), 0.5),
    ],
)
def test_strict_deployment_contract_rejects_metadata_mutations(path: tuple[str, ...], value):
    strict, inventory, topology_gate = strict_contract_fixture()
    assert select.validate_strict_deployment_contract(
        strict, inventory, topology_gate
    )["deployment_eligible"]
    node = strict["configuration"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="Strict deployment contract validation failed"):
        select.validate_strict_deployment_contract(strict, inventory, topology_gate)


def make_minimal_deterministic_evidence(tmp_path: Path, monkeypatch):
    case_id = "subject019"
    name = "deterministic-fixture"
    monkeypatch.setattr(select, "CASES", (case_id,))
    matrix_path = tmp_path / "labels/final-eval/deterministic/matrix.json"
    write_json(matrix_path, {"fixture": True})
    prediction = {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": [],
    }
    runtime = {"runtime_s": 1.0, "peak_rss_mb": 2.0}
    config = select.asdict(select.DetectorConfig())
    record = {
        "name": name,
        "case_id": case_id,
        "matrix_sha256": select.digest(matrix_path),
        "runner_sha256": select.PINNED_DETERMINISTIC_RUNNER_SHA256,
        "prediction": prediction,
        "ordered_candidate_hashes": [],
        "diagnostics": {"config": config},
        "proposal_audit": [{"config": config, "candidates": []}],
        "runtime": runtime,
    }
    receipt_path = tmp_path / f"labels/final-eval/deterministic/runs/{name}/{case_id}.json"
    write_json(receipt_path, record)
    for variant_name in (name, f"{name}-trace-ceiling"):
        write_json(
            tmp_path / f"labels/final-eval/deterministic/predictions/{variant_name}/{case_id}.json",
            prediction,
        )
    failure_record: dict[str, list[object]] = {"failures": []}
    write_json(tmp_path / "labels/final-eval/deterministic/failures.json", failure_record)
    write_json(
        tmp_path / "labels/final-eval/deterministic/checkpoint.json",
        {
            "snapshot_completed_pair_count": 85,
            "snapshot_pending_pair_count": 45,
            "completed_pairs": [{"name": name, "case_id": case_id}],
            "pending_pairs": [],
            "superseded_by": {"completed_pairs": 130, "pending_pairs": 0},
        },
    )
    spec = {
        "name": name,
        "engine": "detect",
        "detector_config": config,
        "strict_union_config": None,
    }
    variants = [
        {
            "family": "deterministic",
            "name": variant_name,
            "predictions": {case_id: prediction},
            "runtime": {case_id: runtime},
        }
        for variant_name in (name, f"{name}-trace-ceiling")
    ]
    report = {"worker_failures": failure_record}
    context = {"specifications": {name: spec}}
    return report, variants, context, receipt_path


def test_deterministic_receipt_set_rejects_missing_file(tmp_path: Path, monkeypatch):
    report, variants, context, receipt_path = make_minimal_deterministic_evidence(tmp_path, monkeypatch)
    select.validate_deterministic_evidence(tmp_path, report, variants, context)
    receipt_path.unlink()
    with pytest.raises(ValueError, match="deterministic receipts.*path set drifted"):
        select.validate_deterministic_evidence(tmp_path, report, variants, context)


def test_deterministic_receipt_replay_rejects_mutation(tmp_path: Path, monkeypatch):
    report, variants, context, receipt_path = make_minimal_deterministic_evidence(tmp_path, monkeypatch)
    record = select.strict_read_json(receipt_path)
    record["runner_sha256"] = "0" * 64
    write_json(receipt_path, record)
    with pytest.raises(ValueError, match="receipt identity drift"):
        select.validate_deterministic_evidence(tmp_path, report, variants, context)


def test_topology_checkpoint_set_rejects_missing_file(tmp_path: Path):
    config = select.strict_read_json(select.ROOT / "labels/final-eval/topology/experiment_config.json")
    case_ids = select.topology_case_ids(config)
    directory = tmp_path / "topology-strict-audit"
    expected = {f"{case_id}.json" for case_id in case_ids}
    for name in expected:
        write_json(directory / name, {})
    select._exact_children(directory, expected, "topology checkpoints for topology-strict-audit")
    (directory / sorted(expected)[0]).unlink()
    with pytest.raises(ValueError, match="topology checkpoints.*path set drifted"):
        select._exact_children(directory, expected, "topology checkpoints for topology-strict-audit")


def test_topology_checkpoint_semantic_replay_rejects_mutation():
    case_id = "synthetic_000_20260913"
    variant = "topology-strict-audit"
    report = select.strict_read_json(select.ROOT / "labels/final-eval/topology/synthetic_report.json")
    reference = next(
        row["reference"] for row in report["cohort"] if row["reference"]["case_id"] == case_id
    )
    record = select.strict_read_json(
        select.ROOT / f"labels/final-eval/topology/synthetic/{variant}/{case_id}.json"
    )
    detector_config = copy.deepcopy(record["audit"]["configuration"])
    root_separation = record["audit"]["root_separation_mm"]
    select.validate_topology_checkpoint(
        record,
        variant=variant,
        case_id=case_id,
        detector_config=detector_config,
        root_separation_mm=root_separation,
        reference=reference,
    )
    record["scores"]["3"]["false_positives"] += 1
    with pytest.raises(ValueError, match="checkpoint score replay drift"):
        select.validate_topology_checkpoint(
            record,
            variant=variant,
            case_id=case_id,
            detector_config=detector_config,
            root_separation_mm=root_separation,
            reference=reference,
        )


def test_real_evidence_replay_has_exact_required_counts():
    reports, _ = select.load_reports()
    _, inventory = select.validate_inventory(reports)
    replay = inventory["deterministic_replay"]
    assert replay["verdict"] == "PASS"
    assert replay["criteria"]["receipt_pairs"] == 130
    assert replay["criteria"]["prediction_pairs"] == 260
    assert len(inventory["evidence_paths"]["deterministic_receipts"]) == 130
    assert len(inventory["evidence_paths"]["deterministic_predictions"]) == 260

    topology = select.validate_topology_gate(select.ROOT)
    assert topology["checkpoint_evidence"]["count"] == 264
    assert len(topology["checkpoint_evidence"]["files"]) == 264
    assert topology["criteria"]["strict_snapshots_tp_fp_fn"] == {
        "2": [45, 1, 4],
        "3": [46, 0, 3],
        "5": [46, 0, 3],
    }
    assert topology["criteria"]["strict_negative_control_predictions"] == 0
