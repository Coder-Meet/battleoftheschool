"""Fail-closed cross-family selection for the five judge-approved reference cases.

The held-out composite estimates the frozen selection procedure.  It is deliberately
separate from the one fixed deployment algorithm and from the all-five development
winner.  No inference or fitting is performed here; every cached prediction is replayed
through :mod:`final_evaluation` before it can enter a ranking.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from detector import DetectorConfig
from evaluate import validate_prediction
from final_evaluation import (
    CASES,
    ROOT,
    TOLERANCES,
    digest,
    score_prediction,
    score_variant,
    summarize,
    write_json,
)
from stress import FAMILIES as STRESS_FAMILIES

OUTPUT = ROOT / "labels/final-eval/selection"
RESULTS_PATH = ROOT / "FINAL_EVALUATION_RESULTS.md"
REPORT_PATHS = {
    "deterministic": Path("labels/final-eval/deterministic/report.json"),
    "tabular": Path("labels/final-eval/tabular/report.json"),
    "cnn": Path("labels/final-eval/cnn/report.json"),
    "topology": Path("labels/final-eval/topology/report.json"),
    "audit": Path("labels/final-eval/audit/report.json"),
}
EXPECTED_VARIANTS = {"deterministic": 52, "tabular": 106, "cnn": 112, "topology": 9, "audit": 1}
EXPECTED_ELIGIBLE = {"deterministic": 12, "tabular": 84, "cnn": 84, "topology": 0, "audit": 0}
TOLERANCE_KEYS = tuple(f"{value:g}" for value in TOLERANCES)
STRICT_ID = "deterministic:deterministic-strict"
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPLICATES = 10_000
PINNED_SCORER_SHA256 = "be2488c1123e0f3ae7cbff7a319ac360fab60c9502ebe5ce490bd7c918d7c990"
PINNED_DETECTOR_SHA256 = "9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e"
PINNED_RELEASE_MANIFEST_SHA256 = "78c2cd3af2c451c60a91e947fd32c740704d02f94d63cd566e0b927d50ce4850"
PINNED_DETERMINISTIC_MATRIX_SHA256 = "fcf4c35fb1aacbd54141a1b09e2d69b9f889b4325c80ad3235c6063108aa8c38"
PINNED_DETERMINISTIC_RUNNER_SHA256 = "69bc364d810bf402a7035e5abccfdf6a360cf6d5643dbda6ebaf5d6274233de1"
PINNED_TOPOLOGY_CONFIG_SHA256 = "db719d635fd57f724cc4ede61edcdc1851d1688073c7c2af96f1da338bb96a94"
DETERMINISTIC_FROZEN_REVISION = "a3cf586031caf67b3f6ef7965ed30d0db717e257"
TOPOLOGY_FROZEN_REVISION = "76ce2f13b09da1f66cbeb6d8dd63dcdd5b00b1db"
PINNED_REFERENCES = {
    "subject019": "38435a327008a660a826c250dc9ec119ef6eae6cee2bd1ca62dd7f59b1a9dbf2",
    "subject020": "dd9f3b2d780f04613daf2e114668c4589b6d4971644f4d64a28cd7056ae874f3",
    "subject021": "9f7e48a9f6be02c032aae5fefd3c77b0376292056eb107ecaad68886ce494029",
    "subject022": "ea72c9717b11d43f9bef7da748698a9ec72575a03f004f762e7f30baddca14d3",
    "subject023": "19172eae21cfab45382a88bcf4bd3cfc4743f97b9234f8a588f918ee2601cb5b",
}
PINNED_INPUTS = {
    "subject019": {
        "image": "5e3140eeaa821b6c964620256c1696df6a228564896b1c66d3846f625b7c8c96",
        "mask": "c18f4c8f43904e1658164e762984d042b1bc88e789494e6fdeec79ec95d82b0f",
    },
    "subject020": {
        "image": "f9b0ff4e3af19ca8e63cddfcc1a5ec5d5d302b1f7d97e54b7d57b06680cc7d81",
        "mask": "dee54f9885040b24bf59e1e11843c4cf08992f12b4ac9479f622a79a279092ca",
    },
    "subject021": {
        "image": "11e3ce094db5632f7f4df2dbe5849923e9f51a7a17eeeeaf1fa6a296abb48519",
        "mask": "0ef0d75f4a912a947ab566da3ef5e614f458f3ee5cba328cf1976b320e7f444b",
    },
    "subject022": {
        "image": "56b888fd4a43396ea38d44c7f3ffa7f4f2972815e8173fa73e47d694f3a3ba81",
        "mask": "96fac7941744573e8e14f0e4402d60d00fdd9099f984df76514b929bd52b3d0d",
    },
    "subject023": {
        "image": "9e2747d40b4f63ec23d4d4a99dd6d4f4cf0125d608c1e782df6f094af51cbb49",
        "mask": "bb10d5cbe26775e76beddf2c3c7d9f447e39784e129fe40f9204e5d48be00cd8",
    },
}
PINNED_INPUT_SIZES = {
    "subject019": {"image": 12_681_000, "mask": 11_634},
    "subject020": {"image": 20_080_961, "mask": 80_497},
    "subject021": {"image": 12_172_260, "mask": 11_615},
    "subject022": {"image": 21_014_843, "mask": 20_884},
    "subject023": {"image": 13_102_536, "mask": 11_323},
}
DETERMINISTIC_SUPPORTING_PATHS = (
    "labels/final-eval/deterministic/matrix.json",
    "labels/final-eval/deterministic/failures.json",
    "labels/final-eval/deterministic/checkpoint.json",
    "labels/final-eval/deterministic/interrupted-source-check-failures.json",
    "labels/final-eval/deterministic/reference-diagnostics.json",
    "labels/final-eval/deterministic/report.json",
    "labels/final-eval/deterministic/HANDOFF.md",
)
TOPOLOGY_SUPPORTING_PATHS = (
    "labels/final-eval/topology/experiment_config.json",
    "labels/final-eval/topology/baseline_report.json",
    "labels/final-eval/topology/report.json",
    "labels/final-eval/topology/synthetic_report.json",
    "labels/final-eval/topology/HANDOFF.md",
)
SELECTOR_SUPPORTING_PATHS = (
    "final_eval_select.py",
    "FINAL_EVALUATION_PROTOCOL.md",
    "research_resources.py",
)

JsonDict = dict[str, Any]
ScoreFunction = Callable[[dict[str, JsonDict]], JsonDict]


def strict_read_json(path: Path) -> JsonDict:
    """Read a JSON object while rejecting JavaScript-style NaN/Infinity constants."""

    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value} in {path}.")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path, root: Path = ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def contained_path(root: Path, name: str, *, directory: bool = False) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Repository artifact path escapes the root: {name}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Repository artifact path escapes the root: {name}")
    if directory and not resolved.is_dir():
        raise ValueError(f"Prediction directory is missing: {name}")
    if not directory and not resolved.is_file():
        raise ValueError(f"Artifact is missing: {name}")
    return resolved


def assert_finite(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite number in {context}.")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{context}[{index}]")


def equivalent(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    """Compare replayed scores while allowing platform-level final-bit reductions."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equivalent(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(equivalent(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def validate_pinned_file_identity(path: Path, expected_sha256: str, expected_size: int) -> JsonDict:
    """Accept exact payload bytes or the canonical Git LFS identity pointer."""
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) == expected_size and actual_sha256 == expected_sha256:
        return {
            "sha256": expected_sha256,
            "size": expected_size,
            "representation": "resolved_payload",
            "payload_resolved": True,
        }
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{expected_sha256}\n"
        f"size {expected_size}\n"
    ).encode("ascii")
    if payload == pointer:
        return {
            "sha256": expected_sha256,
            "size": expected_size,
            "representation": "canonical_git_lfs_pointer",
            "payload_resolved": False,
        }
    if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"Malformed or mismatched Git LFS pointer identity: {path}")
    raise ValueError(f"Resolved payload hash or size drift: {path}")


def validate_repository_pins(root: Path = ROOT) -> JsonDict:
    scorer = contained_path(root, "final_evaluation.py")
    if digest(scorer) != PINNED_SCORER_SHA256:
        raise ValueError("Shared scorer hash drifted from the frozen comparison scorer.")
    detector = contained_path(root, "detector.py")
    if digest(detector) != PINNED_DETECTOR_SHA256:
        raise ValueError("Production detector hash drifted from the evaluated strict baseline.")
    reference_hashes = {}
    for case, expected in PINNED_REFERENCES.items():
        path = contained_path(root, f"labels/organizer-v1/references/{case}.json")
        reference_hashes[case] = digest(path)
        if reference_hashes[case] != expected:
            raise ValueError(f"Reference hash drift for {case}.")
    manifest = contained_path(root, "eval/docs/manifest.json")
    if digest(manifest) != PINNED_RELEASE_MANIFEST_SHA256:
        raise ValueError("Release manifest hash drifted.")
    validation = strict_read_json(contained_path(root, "labels/organizer-v1/validation.json"))
    if validation.get("manifest_sha256") != PINNED_RELEASE_MANIFEST_SHA256:
        raise ValueError("Validation record points to a different release manifest.")
    rows = {row["case_id"]: row for row in validation.get("cases", [])}
    if set(rows) != set(CASES):
        raise ValueError("Validation record must contain exactly the five selection cases.")
    input_hashes = {}
    input_identity = {}
    for case in CASES:
        image_paths = sorted((root / "data" / case).glob("orig*.nii*"))
        mask_paths = sorted((root / "data" / case).glob("mask*.nii*"))
        if len(image_paths) != 1 or len(mask_paths) != 1:
            raise ValueError(f"Need exactly one image and parent mask identity for {case}.")
        input_identity[case] = {
            "image": validate_pinned_file_identity(
                image_paths[0], PINNED_INPUTS[case]["image"], PINNED_INPUT_SIZES[case]["image"]
            ),
            "mask": validate_pinned_file_identity(
                mask_paths[0], PINNED_INPUTS[case]["mask"], PINNED_INPUT_SIZES[case]["mask"]
            ),
        }
        input_hashes[case] = dict(PINNED_INPUTS[case])
        if any(rows[case][f"{kind}_sha256"] != value for kind, value in input_hashes[case].items()):
            raise ValueError(f"Validation/input identity disagreement for {case}.")
    return {
        "scorer_sha256": digest(scorer),
        "detector_sha256": digest(detector),
        "reference_sha256": reference_hashes,
        "input_sha256": input_hashes,
        "input_identity": input_identity,
        "all_input_payloads_resolved": all(
            row[kind]["payload_resolved"] for row in input_identity.values() for kind in ("image", "mask")
        ),
        "selection_replay_requires_resolved_payloads": False,
        "release_manifest_sha256": digest(manifest),
    }


def report_reference_hashes(report: JsonDict) -> JsonDict | None:
    for key in ("reference_hashes", "reference_sha256", "references_sha256"):
        value = report.get(key)
        if isinstance(value, dict):
            return value
    return None


def validate_report_provenance(family: str, report: JsonDict) -> None:
    references = report_reference_hashes(report)
    if references is not None and references != PINNED_REFERENCES:
        raise ValueError(f"{family} report references differ from the pinned references.")
    candidates = []
    if isinstance(report.get("source_hashes"), dict):
        candidates.append(report["source_hashes"].get("final_evaluation.py"))
    if isinstance(report.get("source_sha256"), dict):
        candidates.append(report["source_sha256"].get("final_evaluation.py"))
    candidates.append(report.get("scoring_source_sha256"))
    for value in candidates:
        if value is not None and value != PINNED_SCORER_SHA256:
            raise ValueError(f"{family} report used a different shared scorer.")


def _exact_children(directory: Path, expected: set[str], context: str) -> None:
    if not directory.is_dir():
        raise ValueError(f"Missing {context} directory: {directory}")
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} path set drifted; missing={missing}, extra={extra}.")


def _source_paths(root: Path, hashes: Mapping[str, str], context: str) -> list[str]:
    paths = []
    for name, expected in hashes.items():
        path = contained_path(root, name)
        if digest(path) != expected:
            raise ValueError(f"{context} source hash drift for {name}.")
        paths.append(relative(path, root))
    return paths


def _recorded_paths(root: Path, hashes: Mapping[str, str], context: str) -> list[str]:
    paths = []
    for name, expected in hashes.items():
        path = contained_path(root, name)
        if digest(path) != expected:
            raise ValueError(f"{context} artifact hash drift for {name}.")
        paths.append(relative(path, root))
    return paths


def _validate_excluded_five_training(root: Path, model: JsonDict) -> list[str]:
    report_path = contained_path(root, "labels/final-eval/tabular/training/report.json")
    split_path = contained_path(root, "labels/final-eval/tabular/training/split.json")
    artifact_path = contained_path(root, model["path"])
    report = strict_read_json(report_path)
    split = strict_read_json(split_path)
    artifact = strict_read_json(artifact_path)
    if report.get("reference_outcomes_used_for_fitting") is not False:
        raise ValueError("Excluded-five training used or ambiguously recorded reference outcomes.")
    if report.get("post_reference_refit") is not True:
        raise ValueError("Excluded-five training must disclose its post-reference refit status.")
    if report.get("model_sha256") != digest(artifact_path) or model.get("sha256") != digest(artifact_path):
        raise ValueError("Excluded-five model hash does not match its training report.")
    if report.get("split") != split or artifact.get("split") != split or model.get("split") != split:
        raise ValueError("Excluded-five model, split, audit, and training report disagree.")
    if any(set(values) & set(CASES) for values in split.values()):
        raise ValueError("Excluded-five declared split contains a released identity.")
    exclusions = report.get("exclusions", {})
    if set(exclusions.get("excluded_reference_case_ids", [])) != set(CASES):
        raise ValueError("Excluded-five report does not enumerate all released identities.")
    aliases = exclusions.get("identity_inventory", {})
    if not isinstance(aliases, dict) or not set(CASES).issubset(set(aliases.values())):
        raise ValueError("Excluded-five identity inventory is incomplete.")
    selected = report.get("selected_records")
    if not isinstance(selected, list) or exclusions.get("retained_rows") != len(selected):
        raise ValueError("Excluded-five retained-record accounting is incomplete.")
    removed_fingerprints = {
        row.get("fingerprint_sha256") for row in exclusions.get("removed_rows", []) if isinstance(row, dict)
    }
    for row in selected:
        canonical_case = aliases.get(row.get("case_id"), row.get("case_id"))
        if canonical_case in CASES:
            raise ValueError("Excluded-five selected records contain a released identity.")
        if canonical_sha256(row.get("fingerprint")) in removed_fingerprints:
            raise ValueError("Excluded-five selected records reuse an excluded released-case fingerprint.")
    calibration = report.get("calibration", {})
    if calibration.get("fitting_case_ids") != []:
        raise ValueError("Excluded-five calibration used case outcomes.")
    evidence = [relative(report_path, root), relative(split_path, root), relative(artifact_path, root)]
    corpus = contained_path(root, report.get("corpus_path", ""))
    if digest(corpus) != report.get("corpus_sha256"):
        raise ValueError("Excluded-five corpus hash drifted.")
    historical_split = contained_path(root, "labels/split.json")
    if digest(historical_split) != report.get("historical_split_sha256"):
        raise ValueError("Excluded-five historical split hash drifted.")
    driver = contained_path(root, "final_eval_tabular.py")
    if digest(driver) != report.get("driver_sha256"):
        raise ValueError("Excluded-five producer hash drifted.")
    evidence.extend([relative(corpus, root), relative(historical_split, root), relative(driver, root)])
    evidence.extend(_source_paths(root, report.get("source_sha256", {}), "excluded-five training"))
    return evidence


def _prepare_deterministic(root: Path, report: JsonDict) -> JsonDict:
    matrix_path = contained_path(root, "labels/final-eval/deterministic/matrix.json")
    if digest(matrix_path) != PINNED_DETERMINISTIC_MATRIX_SHA256:
        raise ValueError("Deterministic frozen matrix hash drifted.")
    matrix = strict_read_json(matrix_path)
    if matrix.get("schema_version") != 1 or matrix.get("cases") != list(CASES):
        raise ValueError("Deterministic matrix schema or case set drifted.")
    if matrix.get("tolerances_mm") != list(TOLERANCES):
        raise ValueError("Deterministic matrix tolerances drifted.")
    if matrix.get("runner_at_freeze_sha256") != PINNED_DETERMINISTIC_RUNNER_SHA256:
        raise ValueError("Deterministic frozen runner hash drifted.")
    if matrix.get("source_hashes", {}).get("detector.py") != PINNED_DETECTOR_SHA256:
        raise ValueError("Deterministic matrix used a different detector.")
    if matrix.get("source_hashes", {}).get("final_evaluation.py") != PINNED_SCORER_SHA256:
        raise ValueError("Deterministic matrix used a different scorer.")
    for case in CASES:
        row = matrix.get("input_hashes", {}).get(case, {})
        if {"image": row.get("image_sha256"), "mask": row.get("mask_sha256")} != PINNED_INPUTS[case]:
            raise ValueError(f"Deterministic matrix input identity drift for {case}.")
    if matrix.get("reference_hashes") != PINNED_REFERENCES:
        raise ValueError("Deterministic matrix reference identities drifted.")
    specifications = matrix.get("specifications")
    if not isinstance(specifications, list) or len(specifications) != 26:
        raise ValueError("Deterministic matrix must contain exactly 26 extraction specifications.")
    specs = {row.get("name"): row for row in specifications}
    if len(specs) != 26 or None in specs:
        raise ValueError("Deterministic matrix specification names are missing or duplicated.")
    derived = matrix.get("derived_variants")
    expected_derived = {
        f"{name}-trace-ceiling": {
            "name": f"{name}-trace-ceiling",
            "source": name,
            "operation": "all_successful_traces",
            "eligible_for_selection": False,
        }
        for name in specs
    }
    if not isinstance(derived, list) or {row.get("name"): row for row in derived} != expected_derived:
        raise ValueError("Deterministic trace-ceiling matrix drifted.")
    if report.get("matrix_path") != "labels/final-eval/deterministic/matrix.json":
        raise ValueError("Deterministic report points to a different matrix.")
    if report.get("matrix_sha256") != PINNED_DETERMINISTIC_MATRIX_SHA256:
        raise ValueError("Deterministic report matrix hash drifted.")
    if report.get("runner_sha256") != PINNED_DETERMINISTIC_RUNNER_SHA256:
        raise ValueError("Deterministic report runner hash drifted.")
    if report.get("source_hashes") != matrix.get("source_hashes"):
        raise ValueError("Deterministic report and matrix source hashes disagree.")
    if report.get("input_hashes") != matrix.get("input_hashes"):
        raise ValueError("Deterministic report and matrix input hashes disagree.")
    if report.get("reference_hashes") != matrix.get("reference_hashes"):
        raise ValueError("Deterministic report and matrix reference hashes disagree.")
    return {
        "matrix": matrix,
        "specifications": specs,
        "evidence_paths": [relative(matrix_path, root)],
    }


def _prepare_tabular(root: Path, report: JsonDict) -> JsonDict:
    audit_path = contained_path(root, "labels/final-eval/tabular/model-audit.json")
    training_path = contained_path(root, "labels/final-eval/tabular/training/report.json")
    if report.get("model_audit") != relative(audit_path, root):
        raise ValueError("Tabular report points to a different model audit.")
    if report.get("training_report") != relative(training_path, root):
        raise ValueError("Tabular report points to a different training report.")
    driver = contained_path(root, "final_eval_tabular.py")
    if report.get("driver_sha256") != digest(driver):
        raise ValueError("Tabular report producer hash drifted.")
    source_hashes = report.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("Tabular report lacks source hashes.")
    evidence_paths = [relative(audit_path, root), relative(training_path, root), relative(driver, root)]
    evidence_paths.extend(_source_paths(root, source_hashes, "tabular report"))
    audit = strict_read_json(audit_path)
    rows = audit.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Tabular model audit is empty.")
    models: dict[str, JsonDict] = {}
    derived_eligibility: dict[str, bool] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or name in models:
            raise ValueError("Tabular model audit names are missing or duplicated.")
        path = contained_path(root, row.get("path", ""))
        payload = strict_read_json(path)
        if digest(path) != row.get("sha256"):
            raise ValueError(f"Tabular model artifact hash drift for {name}.")
        expected_kind = "tree" if payload.get("model_type") == "branchseed_candidate_trees" else "logistic"
        for key, expected in (
            ("kind", expected_kind),
            ("threshold", payload.get("threshold")),
            ("feature_names", payload.get("feature_names")),
            ("split", payload.get("split")),
            ("calibration", payload.get("calibration")),
        ):
            if row.get(key) != expected:
                raise ValueError(f"Tabular model audit {key} drift for {name}.")
        split = payload.get("split")
        if not isinstance(split, dict):
            raise ValueError(f"Tabular model {name} lacks a declared split.")
        exposure = {part: sorted(set(values) & set(CASES)) for part, values in split.items()}
        if row.get("reference_exposure_by_partition") != exposure:
            raise ValueError(f"Tabular model audit exposure drift for {name}.")
        evidence_paths.append(relative(path, root))
        lineage_hashes = row.get("lineage_sha256", {})
        if lineage_hashes:
            evidence_paths.extend(_recorded_paths(root, lineage_hashes, f"tabular lineage {name}"))
        current = row["path"].startswith("labels/research/current-source-v1/models/")
        excluded = row["path"] == "labels/final-eval/tabular/training/logistic.json"
        historical = row["path"] in {"labels/candidate-model.json", "labels/candidate-model-synthetic.json"}
        replacement_name = row.get("identical_current_source_artifact")
        cohort_identical = False
        if replacement_name is not None:
            replacement = contained_path(root, replacement_name)
            evidence_paths.append(relative(replacement, root))
            cohort_identical = digest(replacement) == digest(path)
            if not cohort_identical:
                raise ValueError(f"Tabular cohort/current artifact mismatch for {name}.")
        compatible = current or excluded or historical or cohort_identical
        if row.get("compatible") is not compatible:
            raise ValueError(f"Tabular model-audit compatibility contradiction for {name}.")
        if compatible and row.get("failure") != "":
            raise ValueError(f"Compatible tabular model {name} records a failure.")
        if not compatible and not row.get("failure"):
            raise ValueError(f"Incompatible tabular model {name} lacks its rejection evidence.")
        if current and expected_kind == "tree":
            contract = payload.get("metadata", {}).get("inference_contract")
            if row.get("inference_contract") != contract:
                raise ValueError(f"Tabular tree inference contract drift for {name}.")
        if excluded:
            evidence_paths.extend(_validate_excluded_five_training(root, row))
        eligible = compatible and not historical and not any(exposure.values())
        if row.get("eligible_for_selection") is not eligible:
            raise ValueError(f"Tabular model-audit eligibility contradiction for {name}.")
        models[name] = row
        derived_eligibility[name] = eligible
    profiles: dict[str, dict[str, JsonDict]] = {}
    expected_config = {"strict": asdict(DetectorConfig()), "review-union": asdict(DetectorConfig.review())}
    for profile in ("strict", "review-union"):
        directory = root / "labels/final-eval/tabular/candidates" / profile
        _exact_children(directory, {f"{case}.json" for case in CASES}, f"tabular {profile} candidates")
        profiles[profile] = {}
        for case in CASES:
            path = directory / f"{case}.json"
            cache = strict_read_json(path)
            if cache.get("case_id") != case or cache.get("profile") != profile:
                raise ValueError(f"Tabular candidate identity mismatch for {profile}/{case}.")
            if cache.get("input_sha256") != PINNED_INPUTS[case]:
                raise ValueError(f"Tabular candidate input identity mismatch for {profile}/{case}.")
            if cache.get("source_sha256") != source_hashes:
                raise ValueError(f"Tabular candidate source mismatch for {profile}/{case}.")
            if cache.get("driver_sha256") != report.get("driver_sha256"):
                raise ValueError(f"Tabular candidate producer mismatch for {profile}/{case}.")
            if cache.get("detector_config") != expected_config[profile]:
                raise ValueError(f"Tabular candidate detector config mismatch for {profile}/{case}.")
            if cache.get("strict_config") != asdict(DetectorConfig()) or cache.get("failures") != []:
                raise ValueError(f"Tabular candidate strict/failure evidence mismatch for {profile}/{case}.")
            for model_name, model in models.items():
                if model.get("compatible"):
                    score = cache.get("model_scores", {}).get(model_name)
                    if not isinstance(score, dict) or score.get("model_sha256") != model.get("sha256"):
                        raise ValueError(f"Tabular candidate/model lineage mismatch for {model_name}/{case}.")
            profiles[profile][case] = cache
            evidence_paths.append(relative(path, root))
    return {
        "models": models,
        "model_eligible": derived_eligibility,
        "profiles": profiles,
        "source_hashes": source_hashes,
        "driver_sha256": report["driver_sha256"],
        "evidence_paths": sorted(set(evidence_paths)),
    }


def _prepare_cnn(root: Path, report: JsonDict) -> JsonDict:
    inventory_path = contained_path(root, "labels/final-eval/cnn/inventory.json")
    if report.get("inventory_path") != relative(inventory_path, root):
        raise ValueError("CNN report points to a different inventory.")
    if report.get("inventory_sha256") != digest(inventory_path):
        raise ValueError("CNN report inventory hash drifted.")
    inventory = strict_read_json(inventory_path)
    source_hashes = inventory.get("current_source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("CNN inventory lacks current source hashes.")
    evidence_paths = [relative(inventory_path, root)]
    evidence_paths.extend(_source_paths(root, source_hashes, "CNN current source"))
    model_rows: dict[str, JsonDict] = {}
    family_by_path = {
        "labels/research/current-source-v1/cnn-synthetic/model.onnx": "current",
        "labels/research/patch-cnn-retrospective-v1/synthetic/model.onnx": "legacy-synthetic",
        "labels/research/patch-cnn-retrospective-v1/mixed/model.onnx": "legacy-mixed",
    }
    for row in inventory.get("onnx_artifacts", []):
        model_path = contained_path(root, row.get("path", ""))
        family = family_by_path.get(row.get("path"))
        if family is None or family in model_rows:
            raise ValueError("CNN inventory model path is unexpected or duplicated.")
        sidecar_path = model_path.with_suffix(".json")
        training_path = model_path.with_name("training-report.json")
        if digest(model_path) != row.get("sha256"):
            raise ValueError(f"CNN model hash drift for {family}.")
        if digest(sidecar_path) != row.get("sidecar_sha256"):
            raise ValueError(f"CNN sidecar hash drift for {family}.")
        if digest(training_path) != row.get("training_report_sha256"):
            raise ValueError(f"CNN training report hash drift for {family}.")
        sidecar = strict_read_json(sidecar_path)
        training = strict_read_json(training_path)
        if sidecar.get("contract") != row.get("contract") or sidecar.get("split") is None:
            raise ValueError(f"CNN sidecar contract/split drift for {family}.")
        records = training.get("train_records")
        if not isinstance(records, list):
            raise ValueError(f"CNN training records missing for {family}.")
        fit_cases = sorted({record.get("case_id") for record in records})
        declared = {
            part: sorted(set(CASES) & set(values)) for part, values in sidecar["split"].items()
        }
        released_fit = sorted(set(CASES) & set(fit_cases))
        if row.get("fit_cases") != fit_cases or row.get("released_cases_in_fit") != released_fit:
            raise ValueError(f"CNN actual-fit lineage drift for {family}.")
        if row.get("released_cases_in_declared_split") != declared:
            raise ValueError(f"CNN declared-split lineage drift for {family}.")
        source_compatible = row.get("contract", {}).get("source_sha256") == {
            key: source_hashes[key] for key in ("detector.py", "candidate_patches.py")
        }
        if row.get("current_source_compatible") is not source_compatible:
            raise ValueError(f"CNN source-compatibility contradiction for {family}.")
        if family == "current" and (released_fit or any(declared.values()) or not source_compatible):
            raise ValueError("Current CNN lineage contains released cases or incompatible source.")
        model_rows[family] = row
        evidence_paths.extend([
            relative(model_path, root), relative(sidecar_path, root), relative(training_path, root)
        ])
    if set(model_rows) != set(family_by_path.values()) or inventory.get("onnx_count") != 3:
        raise ValueError("CNN inventory must contain exactly the three frozen ONNX families.")
    source_evidence = inventory.get("source_evidence", {})
    current_manifest = contained_path(root, source_evidence.get("current_manifest", ""))
    current_reproduction = contained_path(root, source_evidence.get("current_reproduction_receipt", ""))
    current_training = contained_path(root, source_evidence.get("current_training_report", ""))
    manifest = strict_read_json(current_manifest)
    if digest(current_manifest) != inventory.get("training_manifest_sha256"):
        raise ValueError("CNN synthetic training-manifest hash drifted.")
    if canonical_sha256(manifest) != inventory.get("training_manifest_content_sha256"):
        raise ValueError("CNN synthetic training-manifest content drifted.")
    if manifest.get("source") != "analytic_synthetic_geometry":
        raise ValueError("CNN current training manifest is not synthetic-only.")
    manifest_cases = {row.get("case_id") for row in manifest.get("cases", [])}
    if manifest_cases & set(CASES):
        raise ValueError("CNN current training manifest contains released cases.")
    if digest(current_reproduction) != source_evidence.get("current_reproduction_sha256"):
        raise ValueError("CNN current reproduction receipt hash drifted.")
    if digest(current_training) != model_rows["current"].get("training_report_sha256"):
        raise ValueError("CNN current training report path/hash disagreement.")
    tree_path = contained_path(root, "labels/research/current-source-v1/models/gradient_boosting-base.json")
    blend_path = contained_path(root, "labels/research/current-source-v1/cnn-synthetic/blend.json")
    evidence_paths.extend([
        relative(current_manifest, root), relative(current_reproduction, root), relative(current_training, root),
        relative(tree_path, root), relative(blend_path, root),
    ])
    candidates: dict[tuple[str, str], dict[str, JsonDict]] = {}
    resources: dict[tuple[str, str], dict[str, JsonDict]] = {}
    for family in model_rows:
        for proposals in ("strict", "review-union"):
            directory = root / "labels/final-eval/cnn/candidates" / family / proposals
            expected = {f"{case}.json" for case in CASES} | {f"{case}.resources.json" for case in CASES}
            _exact_children(directory, expected, f"CNN {family}/{proposals} candidates")
            key = (family, proposals)
            candidates[key], resources[key] = {}, {}
            for case in CASES:
                candidate_path = directory / f"{case}.json"
                resource_path = directory / f"{case}.resources.json"
                receipt = strict_read_json(candidate_path)
                resource = strict_read_json(resource_path)
                if (
                    receipt.get("status") != "success"
                    or receipt.get("case_id") != case
                    or receipt.get("family") != family
                    or receipt.get("proposals") != proposals
                ):
                    raise ValueError(f"CNN candidate identity/status mismatch for {family}/{proposals}/{case}.")
                if family == "current" and receipt.get("strict_config") != asdict(DetectorConfig()):
                    raise ValueError(f"CNN strict detector config drift for {family}/{proposals}/{case}.")
                if family == "current" and receipt.get("review_config") != asdict(DetectorConfig.review()):
                    raise ValueError(f"CNN review detector config drift for {family}/{proposals}/{case}.")
                expected_input = {"image": PINNED_INPUTS[case]["image"], "aorta_mask": PINNED_INPUTS[case]["mask"]}
                if receipt.get("input_sha256") != expected_input:
                    raise ValueError(f"CNN candidate input identity mismatch for {family}/{proposals}/{case}.")
                if receipt.get("model_sha256", {}).get("onnx") != model_rows[family].get("sha256"):
                    raise ValueError(f"CNN candidate/model hash mismatch for {family}/{proposals}/{case}.")
                validate_prediction(receipt.get("unfiltered_prediction", {}))
                if receipt["unfiltered_prediction"].get("case_id") != case:
                    raise ValueError(f"CNN unfiltered prediction case mismatch for {family}/{proposals}/{case}.")
                if resource.get("exit_code") != 0 or set(resource.get("cases", {})) != {case}:
                    raise ValueError(f"CNN resource receipt failed or mismatched for {family}/{proposals}/{case}.")
                if candidates[key]:
                    first_receipt = next(iter(candidates[key].values()))
                    for field in ("strict_config", "review_config", "source_sha256", "model_sha256"):
                        if receipt.get(field) != first_receipt.get(field):
                            raise ValueError(f"CNN candidate {field} differs across cases for {family}/{proposals}.")
                candidates[key][case], resources[key][case] = receipt, resource
                evidence_paths.extend([relative(candidate_path, root), relative(resource_path, root)])
    return {
        "inventory": inventory,
        "models": model_rows,
        "candidates": candidates,
        "resources": resources,
        "tree_sha256": digest(tree_path),
        "blend_sha256": digest(blend_path),
        "evidence_paths": sorted(set(evidence_paths)),
    }


def _prepare_topology_family(root: Path, report: JsonDict) -> JsonDict:
    config_path = contained_path(root, "labels/final-eval/topology/experiment_config.json")
    if digest(config_path) != PINNED_TOPOLOGY_CONFIG_SHA256:
        raise ValueError("Topology frozen experiment config hash drifted.")
    config = strict_read_json(config_path)
    if config.get("development_status") != "POST-REFERENCE DEVELOPMENT":
        raise ValueError("Topology config no longer records post-reference development.")
    if config.get("eligible_for_selection") is not False:
        raise ValueError("Topology frozen config must remain selection-ineligible.")
    if config.get("source_hashes", {}).get("detector.py") != PINNED_DETECTOR_SHA256:
        raise ValueError("Topology frozen config used a different detector.")
    if config.get("source_hashes", {}).get("final_evaluation.py") != PINNED_SCORER_SHA256:
        raise ValueError("Topology frozen config used a different scorer.")
    if report.get("development_status") != "POST-REFERENCE DEVELOPMENT":
        raise ValueError("Topology report development status drifted.")
    if report.get("source_hashes") != config.get("source_hashes"):
        raise ValueError("Topology report/config source hashes disagree.")
    if report.get("frozen_config_sha256") != PINNED_TOPOLOGY_CONFIG_SHA256:
        raise ValueError("Topology report/config hash disagreement.")
    if report.get("model_hashes") != {} or report.get("failures") != []:
        raise ValueError("Topology report model/failure state is invalid.")
    if report.get("reference_hashes") != PINNED_REFERENCES:
        raise ValueError("Topology report reference identities drifted.")
    for case in CASES:
        if report.get("input_hashes", {}).get(case) != PINNED_INPUTS[case]:
            raise ValueError(f"Topology report input identity drift for {case}.")
    return {"config": config, "evidence_paths": [relative(config_path, root)]}


def prepare_family_evidence(root: Path, reports: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "deterministic": _prepare_deterministic(root, reports["deterministic"]),
        "tabular": _prepare_tabular(root, reports["tabular"]),
        "cnn": _prepare_cnn(root, reports["cnn"]),
        "topology": _prepare_topology_family(root, reports["topology"]),
        "audit": {"evidence_paths": []},
    }


def _deterministic_eligibility(raw: JsonDict, context: JsonDict) -> tuple[bool, bool]:
    name = raw["name"]
    specs = context["specifications"]
    operation = "final"
    source_name = name
    if name.endswith("-trace-ceiling"):
        operation = "trace-ceiling"
        source_name = name.removesuffix("-trace-ceiling")
    if source_name not in specs:
        raise ValueError(f"Deterministic report row is absent from the frozen matrix: {name}.")
    spec = specs[source_name]
    expected = {
        **{key: value for key, value in spec.items() if key != "name"},
        "operation": operation,
        "source_hashes": context["matrix"]["source_hashes"],
        "threads": 4,
    }
    if raw.get("configuration") != expected:
        raise ValueError(f"Deterministic report configuration contradicts the matrix for {name}.")
    detector = spec.get("detector_config", {})
    origin = (
        operation == "final"
        and spec.get("origin_size_eligibility_disabled") is False
        and detector.get("minimum_origin_diameter_mm") is not None
        and float(detector["minimum_origin_diameter_mm"]) >= 2
    )
    reference_free = (
        spec.get("category") == "frozen_existing_algorithm"
        and spec.get("trained_on_reference_cases") is False
        and spec.get("post_reference_new_algorithm") is False
        and spec.get("model_hashes") == {}
        and spec.get("candidate_threshold") is None
    )
    within_resources = all(
        (row.get("peak_rss_mb") or 0) <= context["matrix"]["worker_rss_limit_mb"]
        for row in raw.get("runtime", {}).values()
    )
    return origin and reference_free and within_resources, origin and reference_free


def _tabular_eligibility(raw: JsonDict, context: JsonDict, root: Path) -> tuple[bool, bool]:
    config = raw["configuration"]
    profile = config.get("profile")
    if profile not in context["profiles"]:
        raise ValueError(f"Unknown tabular proposal profile for {raw['name']}.")
    caches = context["profiles"][profile]
    expected_caches = {
        case: {
            "path": f"labels/final-eval/tabular/candidates/{profile}/{case}.json",
            "sha256": digest(root / f"labels/final-eval/tabular/candidates/{profile}/{case}.json"),
        }
        for case in CASES
    }
    if config.get("candidate_caches") != expected_caches:
        raise ValueError(f"Tabular candidate-cache lineage drift for {raw['name']}.")
    first = caches[CASES[0]]
    stable = {
        "detector_config": first["detector_config"],
        "strict_config": first["strict_config"],
        "source_sha256": context["source_hashes"],
        "threads": 4,
    }
    if any(config.get(key) != value for key, value in stable.items()):
        raise ValueError(f"Tabular configuration/source lineage drift for {raw['name']}.")
    model = config.get("model")
    if model is None:
        if raw["name"] != f"tabular-unfiltered-{profile}" or config.get("threshold") != 0:
            raise ValueError(f"Tabular unfiltered row identity drift for {raw['name']}.")
        eligible = True
    else:
        model_name = model.get("name")
        if model_name not in context["models"] or model != context["models"][model_name]:
            raise ValueError(f"Tabular report/model-audit mismatch for {raw['name']}.")
        expected_name = f"{model_name}-{profile}-t{float(config.get('threshold')):.17g}"
        if raw["name"] != expected_name:
            raise ValueError(f"Tabular row name/configuration mismatch for {raw['name']}.")
        eligible = context["model_eligible"][model_name]
    origin = profile == "strict" and config.get("detector_config") == asdict(DetectorConfig())
    return eligible, origin


def _cnn_eligibility(raw: JsonDict, context: JsonDict, root: Path) -> tuple[bool, bool]:
    config = raw["configuration"]
    family, proposals = config.get("family"), config.get("proposals")
    key = (family, proposals)
    if family not in context["models"] or key not in context["candidates"]:
        raise ValueError(f"CNN row has unknown family/proposal lineage: {raw['name']}.")
    directory = f"labels/final-eval/cnn/candidates/{family}/{proposals}"
    if config.get("candidate_dir") != directory:
        raise ValueError(f"CNN candidate directory drift for {raw['name']}.")
    candidate_hashes = {case: digest(root / directory / f"{case}.json") for case in CASES}
    resource_hashes = {case: digest(root / directory / f"{case}.resources.json") for case in CASES}
    if config.get("candidate_sha256") != candidate_hashes or config.get("resource_sha256") != resource_hashes:
        raise ValueError(f"CNN candidate/resource hash drift for {raw['name']}.")
    first = context["candidates"][key][CASES[0]]
    if config.get("source_sha256") != first.get("source_sha256"):
        raise ValueError(f"CNN source lineage drift for {raw['name']}.")
    if config.get("model_sha256") != first.get("model_sha256"):
        raise ValueError(f"CNN model lineage drift for {raw['name']}.")
    expected_detector_settings = {
        "strict_config": first.get("strict_config"), "review_config": first.get("review_config")
    }
    if config.get("detector_settings") != expected_detector_settings:
        raise ValueError(f"CNN detector settings drift for {raw['name']}.")
    if config.get("post_reference_algorithm") is not False:
        raise ValueError(f"CNN post-reference algorithm flag drift for {raw['name']}.")
    model = context["models"][family]
    eligible = (
        family == "current"
        and model.get("current_source_compatible") is True
        and model.get("released_cases_in_fit") == []
        and all(not values for values in model.get("released_cases_in_declared_split", {}).values())
    )
    origin = eligible and proposals == "strict" and first.get("strict_config") == asdict(DetectorConfig())
    return eligible, origin


def derive_family_eligibility(
    family: str, raw: JsonDict, context: JsonDict, root: Path
) -> tuple[bool, bool]:
    if family == "deterministic":
        return _deterministic_eligibility(raw, context)
    if family == "tabular":
        return _tabular_eligibility(raw, context, root)
    if family == "cnn":
        return _cnn_eligibility(raw, context, root)
    if family == "topology":
        expected = context["config"].get("variants", {}).get(raw["name"])
        if raw.get("configuration") != expected:
            raise ValueError(f"Topology report/config mismatch for {raw['name']}.")
        return False, False
    if family == "audit":
        return False, False
    raise ValueError(f"Unsupported family: {family}")


def _candidate_identity(row: JsonDict) -> str:
    value = {key: row[key] for key in ("ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def project_trace_ceiling(record: JsonDict) -> JsonDict:
    daughters: list[JsonDict] = []
    for group in record.get("proposal_audit", []):
        for candidate in group.get("candidates", []):
            branch = candidate.get("branch")
            if branch is None:
                continue
            daughters.append({
                **{
                    key: branch[key]
                    for key in ("parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz")
                },
                "instance_id": f"trace_{len(daughters) + 1:04d}",
            })
    prediction = {
        "case_id": record["case_id"],
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }
    validate_prediction(prediction)
    return prediction


def validate_deterministic_evidence(
    root: Path, report: JsonDict, variants: Sequence[JsonDict], context: JsonDict
) -> JsonDict:
    specs = context["specifications"]
    expected_cases = {f"{case}.json" for case in CASES}
    runs_root = root / "labels/final-eval/deterministic/runs"
    _exact_children(runs_root, set(specs), "deterministic receipt variants")
    variant_map = {row["name"]: row for row in variants if row["family"] == "deterministic"}
    expected_variant_names = set(specs) | {f"{name}-trace-ceiling" for name in specs}
    if set(variant_map) != expected_variant_names:
        raise ValueError("Deterministic report variant set differs from the frozen matrix.")
    prediction_root = root / "labels/final-eval/deterministic/predictions"
    _exact_children(prediction_root, expected_variant_names, "deterministic prediction variants")
    receipt_paths: list[str] = []
    prediction_paths: list[str] = []
    matrix_sha = digest(root / "labels/final-eval/deterministic/matrix.json")
    for name, spec in specs.items():
        run_dir = runs_root / name
        _exact_children(run_dir, expected_cases, f"deterministic receipts for {name}")
        for derived_name in (name, f"{name}-trace-ceiling"):
            _exact_children(prediction_root / derived_name, expected_cases, f"deterministic predictions for {derived_name}")
        final = variant_map[name]
        trace = variant_map[f"{name}-trace-ceiling"]
        for case in CASES:
            path = run_dir / f"{case}.json"
            record = strict_read_json(path)
            if (
                record.get("name") != name
                or record.get("case_id") != case
                or record.get("matrix_sha256") != matrix_sha
                or record.get("runner_sha256") != PINNED_DETERMINISTIC_RUNNER_SHA256
            ):
                raise ValueError(f"Deterministic receipt identity drift for {name}/{case}.")
            validate_prediction(record.get("prediction", {}))
            if record["prediction"] != final["predictions"][case]:
                raise ValueError(f"Deterministic final prediction/receipt drift for {name}/{case}.")
            if project_trace_ceiling(record) != trace["predictions"][case]:
                raise ValueError(f"Deterministic trace-ceiling projection drift for {name}/{case}.")
            expected_candidate_hashes = [_candidate_identity(row) for row in record["prediction"]["daughters"]]
            if record.get("ordered_candidate_hashes") != expected_candidate_hashes:
                raise ValueError(f"Deterministic candidate ordering/hash drift for {name}/{case}.")
            runtime_config = dict(spec["detector_config"])
            if spec.get("engine") == "paper":
                runtime_config["profile"] = f"paper:{spec['paper_method']}"
            if record.get("diagnostics", {}).get("config") != runtime_config:
                raise ValueError(f"Deterministic diagnostics config drift for {name}/{case}.")
            expected_audit_configs = (
                [spec["strict_union_config"], runtime_config]
                if spec.get("engine") == "union"
                else [runtime_config]
            )
            proposal_audit = record.get("proposal_audit")
            if (
                not isinstance(proposal_audit, list)
                or any(not isinstance(group, dict) for group in proposal_audit)
                or [group.get("config") for group in proposal_audit] != expected_audit_configs
            ):
                raise ValueError(f"Deterministic proposal-audit config drift for {name}/{case}.")
            if record.get("runtime") != final["runtime"][case] or record.get("runtime") != trace["runtime"][case]:
                raise ValueError(f"Deterministic receipt/report runtime drift for {name}/{case}.")
            receipt_paths.append(relative(path, root))
            prediction_paths.extend([
                relative(prediction_root / name / f"{case}.json", root),
                relative(prediction_root / f"{name}-trace-ceiling" / f"{case}.json", root),
            ])
    failure_path = contained_path(root, "labels/final-eval/deterministic/failures.json")
    failure_record = strict_read_json(failure_path)
    if failure_record.get("failures") != [] or report.get("worker_failures") != failure_record:
        raise ValueError("Deterministic worker failure evidence is nonempty or inconsistent.")
    checkpoint = strict_read_json(contained_path(root, "labels/final-eval/deterministic/checkpoint.json"))
    expected_pairs = {(name, case) for name in specs for case in CASES}
    snapshot_pairs = {
        (row.get("name"), row.get("case_id"))
        for key in ("completed_pairs", "pending_pairs") for row in checkpoint.get(key, [])
    }
    if (
        checkpoint.get("snapshot_completed_pair_count") != 85
        or checkpoint.get("snapshot_pending_pair_count") != 45
        or snapshot_pairs != expected_pairs
        or checkpoint.get("superseded_by", {}).get("completed_pairs") != 130
        or checkpoint.get("superseded_by", {}).get("pending_pairs") != 0
    ):
        raise ValueError("Deterministic historical checkpoint accounting drifted.")
    return {
        "criteria": {
            "matrix_specifications": 26,
            "receipt_pairs": 130,
            "prediction_pairs": 260,
            "failures_empty": True,
            "receipt_identity_runtime_and_config_replayed": True,
            "final_and_trace_predictions_replayed": True,
        },
        "verdict": "PASS",
        "receipt_paths": sorted(receipt_paths),
        "prediction_paths": sorted(prediction_paths),
    }


def complexity(family: str, variant: JsonDict) -> JsonDict:
    config = variant["configuration"]
    dependency_ids: list[str] = []
    if family == "tabular" and config.get("model") is not None:
        model = config["model"]
        dependency_ids = [str(model.get("path") or model.get("name") or "tabular-model")]
    elif family == "cnn" and not config.get("unfiltered"):
        weight = config.get("cnn_weight")
        hashes = config.get("model_sha256", {})
        if config.get("frozen_blend") or (weight is not None and 0 < float(weight) < 1):
            dependency_ids = [f"tree:{hashes.get('tree')}", f"onnx:{hashes.get('onnx')}"]
        elif weight == 0:
            dependency_ids = [f"tree:{hashes.get('tree')}"]
        else:
            dependency_ids = [f"onnx:{hashes.get('onnx')}"]
    proposal = str(config.get("profile") or config.get("proposals") or "")
    proposal_passes = 2 if proposal == "review-union" or config.get("engine") == "union" else 1
    paper_component = int(bool(config.get("paper_method")))
    return {
        "effective_model_dependencies": len(dependency_ids),
        "dependency_ids": dependency_ids,
        "proposal_passes": proposal_passes,
        "algorithm_components": proposal_passes + len(dependency_ids) + paper_component,
        "normalization": "Effective model artifacts, then proposal/model component count; runtime is a later tie-breaker.",
    }


def validate_score_shape(scores: JsonDict, global_name: str) -> None:
    if tuple(scores) != TOLERANCE_KEYS and set(scores) != set(TOLERANCE_KEYS):
        raise ValueError(f"{global_name} must contain exactly 2/3/5-mm scores.")
    for tolerance in TOLERANCE_KEYS:
        block = scores[tolerance]
        rows = block.get("cases")
        if not isinstance(rows, list) or len(rows) != len(CASES):
            raise ValueError(f"{global_name} has incomplete {tolerance}-mm case scores.")
        case_ids = [row.get("case_id") for row in rows]
        if len(set(case_ids)) != len(case_ids) or set(case_ids) != set(CASES):
            raise ValueError(f"{global_name} has wrong/duplicate {tolerance}-mm cases.")
        if not isinstance(block.get("summary"), dict):
            raise ValueError(f"{global_name} lacks a {tolerance}-mm pooled summary.")
        assert_finite(block, f"{global_name}.scores.{tolerance}")


def validate_runtime(runtime: Any, global_name: str) -> None:
    if not isinstance(runtime, dict) or set(runtime) != set(CASES):
        raise ValueError(f"{global_name} runtime must contain exactly all five cases.")
    for case, row in runtime.items():
        value = row.get("runtime_s") if isinstance(row, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{global_name} has missing/non-finite runtime for {case}.")
        if value < 0:
            raise ValueError(f"{global_name} has negative runtime for {case}.")
        assert_finite(row, f"{global_name}.runtime.{case}")


def load_prediction_set(root: Path, family: str, variant: JsonDict) -> tuple[dict[str, JsonDict], JsonDict]:
    global_name = f"{family}:{variant['name']}"
    directory = contained_path(root, variant.get("prediction_dir", ""), directory=True)
    names = {path.name for path in directory.glob("*.json")}
    expected_names = {f"{case}.json" for case in CASES}
    if names != expected_names:
        raise ValueError(f"{global_name} prediction directory must contain exactly the five case JSON files.")
    predictions: dict[str, JsonDict] = {}
    byte_hashes = {}
    semantic_hashes = {}
    identifiers = {}
    for case in CASES:
        path = directory / f"{case}.json"
        prediction = strict_read_json(path)
        if prediction.get("case_id") != case:
            raise ValueError(f"{global_name} prediction case ID mismatch for {case}.")
        validate_prediction(prediction)
        assert_finite(prediction, f"{global_name}.prediction.{case}")
        daughter_ids = [row.get("instance_id") for row in prediction.get("daughters", [])]
        if len(daughter_ids) != len(set(daughter_ids)) or any(not value for value in daughter_ids):
            raise ValueError(f"{global_name} has missing/duplicate prediction IDs for {case}.")
        predictions[case] = prediction
        byte_hashes[case] = digest(path)
        semantic_hashes[case] = canonical_sha256(prediction)
        identifiers[case] = daughter_ids
    recorded = variant.get("prediction_hashes")
    if recorded is not None and recorded != byte_hashes:
        raise ValueError(f"{global_name} recorded prediction hashes drifted.")
    return predictions, {
        "directory": relative(directory, root),
        "byte_sha256": byte_hashes,
        "semantic_sha256": semantic_hashes,
        "set_semantic_sha256": canonical_sha256({case: predictions[case] for case in CASES}),
        "daughter_ids": identifiers,
    }


def validate_inventory(
    reports: Mapping[str, JsonDict],
    root: Path = ROOT,
    *,
    scorer: ScoreFunction = score_variant,
    expected_variants: Mapping[str, int] = EXPECTED_VARIANTS,
    expected_eligible: Mapping[str, int] | None = EXPECTED_ELIGIBLE,
    require_repository_pins: bool = True,
    fixture_mode: bool = False,
) -> tuple[list[JsonDict], JsonDict]:
    """Normalize/replay every variant and derive production eligibility from evidence."""
    if set(reports) != set(expected_variants):
        raise ValueError("Every declared family report is required exactly once.")
    if fixture_mode and require_repository_pins:
        raise ValueError("Fixture mode cannot be combined with production repository-pin validation.")
    pins = validate_repository_pins(root) if require_repository_pins else {}
    evidence = {} if fixture_mode else prepare_family_evidence(root, reports)
    normalized = []
    seen_names: set[str] = set()
    family_inventory = {}
    family_predictions: dict[str, list[str]] = defaultdict(list)
    for family in expected_variants:
        report = reports[family]
        if report.get("family") != family:
            raise ValueError(f"Expected {family} report, got {report.get('family')!r}.")
        if family != "topology" and report.get("schema_version") != 1:
            raise ValueError(f"{family} report schema version is not 1.")
        if family == "topology" and report.get("schema_version") not in (None, 1):
            raise ValueError("Unsupported topology report schema.")
        validate_report_provenance(family, report)
        variants = report.get("variants")
        if not isinstance(variants, list) or len(variants) != expected_variants[family]:
            raise ValueError(f"{family} must contain exactly {expected_variants[family]} variants.")
        failures = report.get("failures")
        if not isinstance(failures, list):
            raise ValueError(f"{family} report must expose its failure list.")
        if failures:
            raise ValueError(f"{family} contains unresolved failures; selection is withheld.")
        if family == "deterministic" and report.get("worker_failures", {}).get("failures") != []:
            raise ValueError("Deterministic worker failures are unresolved.")
        derived_count = 0
        for raw in variants:
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{family} contains an unnamed variant.")
            global_name = f"{family}:{name}"
            if name in seen_names:
                raise ValueError(f"Variant name is not globally unique: {name}")
            seen_names.add(name)
            if not isinstance(raw.get("configuration"), dict):
                raise ValueError(f"{global_name} lacks configuration provenance.")
            if not isinstance(raw.get("eligible_for_selection"), bool):
                raise ValueError(f"{global_name} has no explicit eligibility boolean.")
            if fixture_mode:
                derived_eligible = raw["eligible_for_selection"]
                derived_origin = derived_eligible and family in {"deterministic", "tabular", "cnn"}
            else:
                derived_eligible, derived_origin = derive_family_eligibility(
                    family, raw, evidence[family], root
                )
                if raw["eligible_for_selection"] is not derived_eligible:
                    raise ValueError(
                        f"{global_name} report eligibility contradicts reconstructed family evidence."
                    )
            derived_count += int(derived_eligible)
            validate_score_shape(raw.get("scores", {}), global_name)
            validate_runtime(raw.get("runtime"), global_name)
            predictions, prediction_provenance = load_prediction_set(root, family, raw)
            replayed = scorer(predictions)
            validate_score_shape(replayed, global_name)
            if not equivalent(replayed, raw["scores"]):
                raise ValueError(f"Cached score drift for {global_name}.")
            clean = derived_eligible and family in {"deterministic", "tabular", "cnn"}
            family_predictions[family].extend(
                f"{prediction_provenance['directory']}/{case}.json" for case in CASES
            )
            normalized.append({
                "family": family,
                "name": name,
                "global_name": global_name,
                "configuration": raw["configuration"],
                "report_eligible": raw["eligible_for_selection"],
                "derived_eligible": derived_eligible,
                "clean_frozen_selection_eligible": clean,
                "origin_policy_certified": derived_origin,
                "deployment_eligible": False,
                "ineligible_reason": raw.get("ineligible_reason", ""),
                "scores": replayed,
                "runtime": raw["runtime"],
                "complexity": complexity(family, raw),
                "predictions": predictions,
                "prediction_provenance": prediction_provenance,
            })
        if expected_eligible is not None and derived_count != expected_eligible[family]:
            raise ValueError(f"{family} eligible count drifted from {expected_eligible[family]}.")
        family_inventory[family] = {"variants": len(variants), "report_eligible": derived_count}
    inventory: JsonDict = {
        "pins": pins,
        "families": family_inventory,
        "fixture_mode": fixture_mode,
        "evidence_paths": {},
    }
    if not fixture_mode:
        deterministic_replay = validate_deterministic_evidence(
            root, reports["deterministic"], normalized, evidence["deterministic"]
        )
        inventory["deterministic_replay"] = {
            "criteria": deterministic_replay["criteria"], "verdict": deterministic_replay["verdict"]
        }
        inventory["deterministic_matrix_source_hashes"] = evidence["deterministic"]["matrix"]["source_hashes"]
        inventory["strict_frozen_pins_validated"] = (
            pins.get("detector_sha256") == PINNED_DETECTOR_SHA256
            and pins.get("scorer_sha256") == PINNED_SCORER_SHA256
            and evidence["deterministic"]["matrix"].get("runner_at_freeze_sha256")
            == PINNED_DETERMINISTIC_RUNNER_SHA256
            and digest(root / "labels/final-eval/deterministic/matrix.json")
            == PINNED_DETERMINISTIC_MATRIX_SHA256
        )
        inventory["evidence_paths"] = {
            family: context.get("evidence_paths", []) for family, context in evidence.items()
        }
        inventory["evidence_paths"]["deterministic_receipts"] = deterministic_replay["receipt_paths"]
        inventory["evidence_paths"]["deterministic_predictions"] = deterministic_replay["prediction_paths"]
        for family, paths in family_predictions.items():
            inventory["evidence_paths"][f"{family}_predictions"] = sorted(set(paths))
    return normalized, inventory


def subset_summary(variant: JsonDict, cases: Sequence[str], tolerance: str = "3") -> JsonDict:
    wanted = set(cases)
    rows = [row for row in variant["scores"][tolerance]["cases"] if row["case_id"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError(f"Incomplete score rows for {variant['global_name']}.")
    return summarize(rows)


def rank_key(variant: JsonDict, cases: Sequence[str]) -> tuple[float, float, int, int, float, str]:
    summary = subset_summary(variant, cases)
    if summary["f1"] is None or summary["count_mae"] is None:
        raise ValueError(f"Undefined ranking metric for {variant['global_name']}.")
    measured_time = sum(float(variant["runtime"][case]["runtime_s"]) for case in cases)
    complexity_row = variant["complexity"]
    return (
        -float(summary["f1"]),
        float(summary["count_mae"]),
        int(complexity_row["effective_model_dependencies"]),
        int(complexity_row["algorithm_components"]),
        measured_time,
        variant["global_name"],
    )


def rank_details(variant: JsonDict, cases: Sequence[str]) -> JsonDict:
    key = rank_key(variant, cases)
    return {
        "pooled_f1_3mm": -key[0],
        "count_mae_3mm": key[1],
        "effective_model_dependencies": key[2],
        "algorithm_components": key[3],
        "measured_runtime_s": key[4],
        "global_name": key[5],
    }


def ranked_entry(variant: JsonDict, cases: Sequence[str]) -> JsonDict:
    return {
        "global_name": variant["global_name"],
        "family": variant["family"],
        "name": variant["name"],
        "rank_key": rank_details(variant, cases),
        "scores_3mm": subset_summary(variant, cases),
        "complexity": variant["complexity"],
        "origin_policy_certified": variant["origin_policy_certified"],
        "deployment_eligible": variant["deployment_eligible"],
        "semantic_prediction_set_sha256": variant["prediction_provenance"]["set_semantic_sha256"],
        "ineligible_reason": variant["ineligible_reason"],
    }


def select_loco(variants: Sequence[JsonDict]) -> tuple[JsonDict, dict[str, JsonDict]]:
    eligible = [row for row in variants if row["clean_frozen_selection_eligible"]]
    if not eligible:
        raise ValueError("No clean frozen candidates are available.")
    folds = []
    held_out_predictions = {}
    for held_out in CASES:
        selection_cases = tuple(case for case in CASES if case != held_out)
        chosen = min(eligible, key=lambda row: rank_key(row, selection_cases))
        held_out_predictions[held_out] = chosen["predictions"][held_out]
        held_out_scores = {
            tolerance: next(
                row for row in chosen["scores"][tolerance]["cases"] if row["case_id"] == held_out
            )
            for tolerance in TOLERANCE_KEYS
        }
        folds.append({
            "held_out_case": held_out,
            "selection_cases": list(selection_cases),
            "selected_global_name": chosen["global_name"],
            "selection_rank_key": rank_details(chosen, selection_cases),
            "selected_configuration": chosen["configuration"],
            "selected_complexity": chosen["complexity"],
            "selected_origin_policy_certified": chosen["origin_policy_certified"],
            "held_out_runtime_s_not_used_for_selection": chosen["runtime"][held_out]["runtime_s"],
            "held_out_source_prediction": {
                "path": f"{chosen['prediction_provenance']['directory']}/{held_out}.json",
                "byte_sha256": chosen["prediction_provenance"]["byte_sha256"][held_out],
                "semantic_sha256": chosen["prediction_provenance"]["semantic_sha256"][held_out],
            },
            "held_out_scores": held_out_scores,
        })
    return {
        "rule": "Other-four pooled 3-mm F1, count MAE, effective model dependencies, algorithm components, "
        "other-four measured runtime, then globally stable lexical name.",
        "eligible_variants": len(eligible),
        "folds": folds,
        "selection_stability": dict(Counter(row["selected_global_name"] for row in folds)),
        "interpretation": "Five whole-case leave-one-out folds of fixed selection; not a deployable mixture.",
    }, held_out_predictions


def signed_count_summary(scores: JsonDict) -> JsonDict:
    result = {}
    for tolerance in TOLERANCE_KEYS:
        values = [row["daughter_counts"]["signed_error"] for row in scores[tolerance]["cases"]]
        result[tolerance] = {
            "per_case": dict(zip(CASES, values)),
            "sum": int(sum(values)),
            "mean": float(np.mean(values)),
        }
    return result


def reference_changes(left: JsonDict, right: JsonDict) -> JsonDict:
    """Return reference IDs gained/lost by left relative to right."""
    output = {}
    for tolerance in TOLERANCE_KEYS:
        rows = []
        for left_row, right_row in zip(left[tolerance]["cases"], right[tolerance]["cases"]):
            if left_row["case_id"] != right_row["case_id"]:
                raise ValueError("Reference comparison case order differs.")
            left_ids = {row["reference_id"] for row in left_row["matches"]}
            right_ids = {row["reference_id"] for row in right_row["matches"]}
            rows.append({
                "case_id": left_row["case_id"],
                "gained_reference_ids": sorted(left_ids - right_ids),
                "lost_reference_ids": sorted(right_ids - left_ids),
            })
        output[tolerance] = rows
    return output


def _bootstrap_estimates(scores: JsonDict, indices: np.ndarray, tolerance: str) -> tuple[np.ndarray, np.ndarray]:
    rows = scores[tolerance]["cases"]
    counts = np.asarray([
        [row["true_positives"], row["false_positives"], row["false_negatives"]] for row in rows
    ])
    totals = counts[indices].sum(axis=1)
    tp, fp, fn = totals.T
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    absolute = np.asarray([row["daughter_counts"]["absolute_error"] for row in rows], dtype=float)
    return f1, absolute[indices].mean(axis=1)


def paired_bootstrap(
    left: JsonDict,
    right: JsonDict,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> JsonDict:
    for tolerance in TOLERANCE_KEYS:
        left_cases = [row["case_id"] for row in left[tolerance]["cases"]]
        right_cases = [row["case_id"] for row in right[tolerance]["cases"]]
        if left_cases != list(CASES) or left_cases != right_cases:
            raise ValueError("Bootstrap must use identical ordered whole cases.")
    indices = np.random.default_rng(seed).integers(0, len(CASES), size=(replicates, len(CASES)))
    results = {}
    for tolerance in TOLERANCE_KEYS:
        left_f1, left_mae = _bootstrap_estimates(left, indices, tolerance)
        right_f1, right_mae = _bootstrap_estimates(right, indices, tolerance)
        results[tolerance] = {
            "left_f1_95_percentile": np.quantile(left_f1, [0.025, 0.975]).tolist(),
            "paired_f1_delta_95_percentile": np.quantile(left_f1 - right_f1, [0.025, 0.975]).tolist(),
            "paired_count_mae_delta_95_percentile": np.quantile(
                left_mae - right_mae, [0.025, 0.975]
            ).tolist(),
        }
    return {
        "unit": "whole case",
        "seed": seed,
        "replicates": replicates,
        "method": "Paired percentile intervals with fixed decisions and no within-resample reselection.",
        "limitation": "Only five reused development cases; intervals do not remove selection or inspection bias.",
        "tolerances": results,
    }


def semantic_aliases(variants: Sequence[JsonDict]) -> list[JsonDict]:
    groups: dict[str, list[JsonDict]] = defaultdict(list)
    for variant in variants:
        groups[variant["prediction_provenance"]["set_semantic_sha256"]].append(variant)
    aliases = []
    for fingerprint, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        eligibility = {row["clean_frozen_selection_eligible"] for row in rows}
        origin = {row["origin_policy_certified"] for row in rows}
        aliases.append({
            "semantic_prediction_set_sha256": fingerprint,
            "global_names": sorted(row["global_name"] for row in rows),
            "clean_eligibility_conflict": len(eligibility) > 1,
            "origin_policy_conflict": len(origin) > 1,
            "note": "Aliases remain provenance-distinct candidates; deterministic rank tie-breakers prevent order effects.",
        })
    return aliases


def resource_summary(variant: JsonDict, cases: Sequence[str] = CASES) -> JsonDict:
    runtimes = [float(variant["runtime"][case]["runtime_s"]) for case in cases]
    rss = [
        float(variant["runtime"][case]["peak_rss_mb"])
        for case in cases
        if variant["runtime"][case].get("peak_rss_mb") is not None
    ]
    return {
        "global_name": variant["global_name"],
        "cases": list(cases),
        "runtime_s": {"sum": sum(runtimes), "mean": float(np.mean(runtimes)), "range": [min(runtimes), max(runtimes)]},
        "maximum_peak_rss_mb": max(rss) if rss else None,
    }


def compact_exclusion(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return str(value)
    keys = ("name", "path", "artifacts", "reason", "failure", "ineligible_reason", "retraining_path")
    return {key: value[key] for key in keys if key in value}


def topology_case_ids(config: JsonDict) -> list[str]:
    cohort = config.get("synthetic", {})
    families = cohort.get("stress_families")
    seeds = cohort.get("stress_seeds")
    scenarios = cohort.get("simple_scenarios")
    simple_seed = cohort.get("simple_seed")
    if not isinstance(families, list) or not isinstance(seeds, list) or not isinstance(scenarios, list):
        raise ValueError("Topology frozen synthetic settings are incomplete.")
    if len(families) != len(set(families)) or any(family not in STRESS_FAMILIES for family in families):
        raise ValueError("Topology frozen stress-family settings are invalid.")
    ids = [
        f"stress_{STRESS_FAMILIES.index(family):02d}_{family}_{seed}"
        for seed in seeds for family in families
    ]
    ids.extend(f"synthetic_{int(scenario):03d}_{simple_seed}" for scenario in scenarios)
    if len(ids) != 24 or len(set(ids)) != 24:
        raise ValueError("Topology frozen settings must derive exactly 24 unique cases.")
    return ids


def validate_topology_checkpoint(
    record: JsonDict,
    *,
    variant: str,
    case_id: str,
    detector_config: JsonDict,
    root_separation_mm: float,
    reference: JsonDict,
) -> tuple[JsonDict, float]:
    if set(record) != {"prediction", "audit", "scores"}:
        raise ValueError(f"Topology checkpoint schema drift for {variant}/{case_id}.")
    prediction = record["prediction"]
    audit = record["audit"]
    if prediction.get("case_id") != case_id:
        raise ValueError(f"Topology checkpoint case mismatch for {variant}/{case_id}.")
    validate_prediction(prediction)
    if audit.get("configuration") != detector_config:
        raise ValueError(f"Topology checkpoint config drift for {variant}/{case_id}.")
    if float(audit.get("root_separation_mm", math.nan)) != float(root_separation_mm):
        raise ValueError(f"Topology checkpoint root separation drift for {variant}/{case_id}.")
    runtime = audit.get("runtime_s")
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not math.isfinite(float(runtime)):
        raise ValueError(f"Topology checkpoint runtime missing/non-finite for {variant}/{case_id}.")
    if float(runtime) < 0:
        raise ValueError(f"Topology checkpoint runtime is negative for {variant}/{case_id}.")
    replayed = {
        tolerance: score_prediction(prediction, reference, float(tolerance))
        for tolerance in TOLERANCE_KEYS
    }
    if set(record["scores"]) != set(TOLERANCE_KEYS) or not equivalent(replayed, record["scores"]):
        raise ValueError(f"Topology checkpoint score replay drift for {variant}/{case_id}.")
    return replayed, float(runtime)


def validate_topology_gate(root: Path) -> JsonDict:
    config_path = contained_path(root, "labels/final-eval/topology/experiment_config.json")
    baseline_path = contained_path(root, "labels/final-eval/topology/baseline_report.json")
    real_report_path = contained_path(root, "labels/final-eval/topology/report.json")
    synthetic_path = contained_path(root, "labels/final-eval/topology/synthetic_report.json")
    if digest(config_path) != PINNED_TOPOLOGY_CONFIG_SHA256:
        raise ValueError("Topology frozen config hash drifted.")
    config = strict_read_json(config_path)
    baseline = strict_read_json(baseline_path)
    real_report = strict_read_json(real_report_path)
    report = strict_read_json(synthetic_path)
    frozen_sources = config.get("source_hashes", {})
    if baseline.get("source_hashes") != frozen_sources:
        raise ValueError("Topology baseline/config source hashes disagree.")
    if real_report.get("source_hashes") != frozen_sources or report.get("source_hashes") != frozen_sources:
        raise ValueError("Topology report/config source hashes disagree.")
    if digest(baseline_path) != config.get("initial_diagnosis_sha256"):
        raise ValueError("Topology initial-diagnosis hash drifted.")
    if real_report.get("frozen_config_sha256") != PINNED_TOPOLOGY_CONFIG_SHA256:
        raise ValueError("Topology real report points to a different frozen config.")
    if report.get("frozen_config_sha256") != PINNED_TOPOLOGY_CONFIG_SHA256:
        raise ValueError("Topology synthetic report points to a different frozen config.")
    if baseline.get("failures") != [] or real_report.get("failures") != [] or report.get("failures") != []:
        raise ValueError("Topology evidence contains failures.")
    current_source_state = {}
    for name, expected in frozen_sources.items():
        path = contained_path(root, name)
        actual = digest(path)
        if name != "research_resources.py" and actual != expected:
            raise ValueError(f"Topology frozen source hash drift for {name}.")
        current_source_state[name] = {
            "frozen_sha256": expected,
            "current_sha256": actual,
            "matches_frozen": actual == expected,
        }
    resource_state = current_source_state.get("research_resources.py", {})
    if resource_state.get("matches_frozen") is not False:
        raise ValueError("Expected documented portability-only research_resources.py drift is absent.")
    case_ids = topology_case_ids(config)
    cohort = report.get("cohort")
    if not isinstance(cohort, list) or [row.get("reference", {}).get("case_id") for row in cohort] != case_ids:
        raise ValueError("Topology synthetic cohort does not match frozen settings/order.")
    references = {}
    negative_controls = []
    for row, case_id in zip(cohort, case_ids):
        reference = row.get("reference")
        provenance = row.get("provenance")
        if not isinstance(reference, dict) or not isinstance(provenance, dict):
            raise ValueError(f"Topology cohort record is incomplete for {case_id}.")
        if provenance.get("case_id") != case_id:
            raise ValueError(f"Topology cohort provenance mismatch for {case_id}.")
        validate_prediction(reference)
        references[case_id] = reference
        if provenance.get("family") == "negative_controls_only":
            negative_controls.append(case_id)
    if len(negative_controls) != 2:
        raise ValueError("Topology cohort must contain exactly two negative controls.")
    strict_setting = {"detector": asdict(DetectorConfig()), "alternate_root_separation_mm": 0.0}
    review_setting = {
        "detector": asdict(DetectorConfig.review(minimum_origin_diameter_mm=2)),
        "alternate_root_separation_mm": 0.0,
    }
    settings = {
        "topology-strict-audit": strict_setting,
        "topology-review-origin2-audit": review_setting,
        **config.get("variants", {}),
    }
    if len(settings) != 11:
        raise ValueError("Topology gate must define exactly 11 variants.")
    synthetic_root = root / "labels/final-eval/topology/synthetic"
    _exact_children(synthetic_root, set(settings), "topology synthetic variants")
    aggregate_rows = report.get("variants")
    if not isinstance(aggregate_rows, list) or {row.get("name") for row in aggregate_rows} != set(settings):
        raise ValueError("Topology synthetic aggregate variant set drifted.")
    aggregate_by_name = {row["name"]: row for row in aggregate_rows}
    checkpoint_files = {}
    rows_for_output = []
    strict_negative_predictions = 0
    for name, setting in settings.items():
        directory = synthetic_root / name
        _exact_children(directory, {f"{case}.json" for case in case_ids}, f"topology checkpoints for {name}")
        score_rows: dict[str, list[JsonDict]] = {tolerance: [] for tolerance in TOLERANCE_KEYS}
        runtimes = {}
        for case_id in case_ids:
            path = directory / f"{case_id}.json"
            record = strict_read_json(path)
            replayed, runtime = validate_topology_checkpoint(
                record,
                variant=name,
                case_id=case_id,
                detector_config=setting["detector"],
                root_separation_mm=setting["alternate_root_separation_mm"],
                reference=references[case_id],
            )
            for tolerance in TOLERANCE_KEYS:
                score_rows[tolerance].append(replayed[tolerance])
            runtimes[case_id] = runtime
            if name == "topology-strict-audit" and case_id in negative_controls:
                strict_negative_predictions += len(record["prediction"]["daughters"])
            checkpoint_files[relative(path, root)] = digest(path)
        aggregate = aggregate_by_name[name]
        expected_scores = {tolerance: summarize(score_rows[tolerance]) for tolerance in TOLERANCE_KEYS}
        if aggregate.get("completed_cases") != 24:
            raise ValueError(f"Topology aggregate case count drift for {name}.")
        if not equivalent(expected_scores, aggregate.get("scores")):
            raise ValueError(f"Topology aggregate score drift for {name}.")
        if aggregate.get("runtime_s") != runtimes:
            raise ValueError(f"Topology aggregate runtime drift for {name}.")
        rows_for_output.append({"name": name, "scores_3mm": expected_scores["3"]})
    strict = aggregate_by_name["topology-strict-audit"]
    snapshots = {
        "2": (45, 1, 4),
        "3": (46, 0, 3),
        "5": (46, 0, 3),
    }
    for tolerance, expected in snapshots.items():
        summary = strict["scores"][tolerance]
        observed = tuple(summary[key] for key in ("true_positives", "false_positives", "false_negatives"))
        if observed != expected:
            raise ValueError(f"Topology strict frozen snapshot drift at {tolerance} mm.")
    if strict_negative_predictions != 0:
        raise ValueError("Topology strict produced detections on negative_controls_only cases.")
    baseline_variants = {row.get("name"): row for row in baseline.get("variants", [])}
    if set(baseline_variants) != {"topology-strict-audit", "topology-review-origin2-audit"}:
        raise ValueError("Topology baseline variant set drifted.")
    if baseline_variants["topology-strict-audit"].get("configuration") != asdict(DetectorConfig()):
        raise ValueError("Topology baseline strict config is not the exact production default.")
    real_variants = {row.get("name"): row for row in real_report.get("variants", [])}
    if set(real_variants) != set(config.get("variants", {})):
        raise ValueError("Topology real recovery variant set drifted.")
    for name, setting in config["variants"].items():
        if real_variants[name].get("configuration") != setting or real_variants[name].get("eligible_for_selection") is not False:
            raise ValueError(f"Topology real report/config eligibility drift for {name}.")
    all_real_names = list(baseline_variants) + list(real_variants)
    real_prediction_paths = []
    diagnosis_paths = []
    resources_paths: list[str] = []
    diagnosis_root = root / "labels/final-eval/topology/diagnosis"
    resources_root = root / "labels/final-eval/topology/resources"
    _exact_children(diagnosis_root, set(all_real_names), "topology diagnosis variants")
    _exact_children(resources_root, set(real_variants), "topology resource variants")
    for name in all_real_names:
        directory = root / "labels/final-eval/topology" / name
        _exact_children(directory, {f"{case}.json" for case in CASES}, f"topology real predictions for {name}")
        diagnosis_dir = diagnosis_root / name
        _exact_children(diagnosis_dir, {f"{case}.json" for case in CASES}, f"topology diagnoses for {name}")
        for case in CASES:
            real_prediction_paths.append(relative(directory / f"{case}.json", root))
            diagnosis_paths.append(relative(diagnosis_dir / f"{case}.json", root))
        if name in real_variants:
            resource_dir = resources_root / name
            _exact_children(resource_dir, {f"{case}.json" for case in CASES}, f"topology resources for {name}")
            resources_paths.extend(relative(resource_dir / f"{case}.json", root) for case in CASES)
    return {
        "path": relative(synthetic_path, root),
        "sha256": digest(synthetic_path),
        "frozen_source_revision": TOPOLOGY_FROZEN_REVISION,
        "current_main_raw_rerun": "intentionally_rejected_by_frozen_source_guard",
        "current_source_state": current_source_state,
        "cases_per_variant": 24,
        "variants": rows_for_output,
        "strict_control": next(row for row in rows_for_output if row["name"] == "topology-strict-audit"),
        "criteria": {
            "exact_checkpoint_set": "11 variants x 24 frozen-derived cases = 264",
            "checkpoint_case_config_root_runtime_replayed": True,
            "scores_recomputed_at_mm": [2, 3, 5],
            "aggregate_and_runtime_maps_match": True,
            "strict_detector_config_equals_default": True,
            "strict_snapshots_tp_fp_fn": {key: list(value) for key, value in snapshots.items()},
            "strict_negative_control_predictions": strict_negative_predictions,
            "failures_empty": True,
        },
        "checkpoint_evidence": {
            "count": len(checkpoint_files),
            "root_sha256": canonical_sha256(checkpoint_files),
            "files": checkpoint_files,
        },
        "supporting_evidence_paths": {
            "real_predictions": sorted(real_prediction_paths),
            "diagnoses": sorted(diagnosis_paths),
            "resources": sorted(resources_paths),
        },
        "verdict": "PASS: exact frozen strict regression criteria satisfied; recovery variants remain diagnostic/ineligible.",
    }


def validate_strict_deployment_contract(
    strict: JsonDict, inventory: JsonDict, topology_gate: JsonDict
) -> JsonDict:
    expected = {
        "engine": "detect",
        "detector_config": asdict(DetectorConfig()),
        "paper_method": "",
        "model_hashes": {},
        "strict_union_config": None,
        "category": "frozen_existing_algorithm",
        "trained_on_reference_cases": False,
        "post_reference_new_algorithm": False,
        "origin_size_eligibility_disabled": False,
        "candidate_threshold": None,
        "operation": "final",
        "source_hashes": inventory["deterministic_matrix_source_hashes"],
        "threads": 4,
    }
    criteria = {
        "exact_full_detector_default": strict.get("configuration", {}).get("detector_config") == asdict(DetectorConfig()),
        "exact_frozen_specification": strict.get("configuration") == expected,
        "final_detect_operation": strict.get("configuration", {}).get("operation") == "final"
        and strict.get("configuration", {}).get("engine") == "detect",
        "no_model_training_or_post_reference_flags": strict.get("configuration", {}).get("model_hashes") == {}
        and strict.get("configuration", {}).get("candidate_threshold") is None
        and strict.get("configuration", {}).get("trained_on_reference_cases") is False
        and strict.get("configuration", {}).get("post_reference_new_algorithm") is False,
        "pinned_detector_scorer_matrix_runner": inventory.get("strict_frozen_pins_validated") is True,
        "deterministic_receipts_replayed": inventory.get("deterministic_replay", {}).get("verdict") == "PASS",
        "topology_strict_gate_replayed": topology_gate.get("criteria", {}).get("exact_checkpoint_set")
        == "11 variants x 24 frozen-derived cases = 264",
        "topology_verdict_passed": str(topology_gate.get("verdict", "")).startswith("PASS:"),
    }
    passed = all(criteria.values())
    if not passed:
        failed = [name for name, value in criteria.items() if not value]
        raise ValueError(f"Strict deployment contract validation failed: {failed}.")
    return {
        "criteria": criteria,
        "origin_policy_certified": passed,
        "deployment_eligible": passed,
        "verdict": "PASS: production-exact deterministic strict contract and raw synthetic gate replayed.",
    }


def find_variant(variants: Sequence[JsonDict], global_name: str) -> JsonDict:
    matches = [row for row in variants if row["global_name"] == global_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {global_name} variant.")
    return matches[0]


def family_records(reports: Mapping[str, JsonDict], key: str) -> JsonDict:
    return {family: report.get(key, []) for family, report in reports.items()}


def build_evaluation(
    reports: Mapping[str, JsonDict],
    variants: Sequence[JsonDict],
    inventory: JsonDict,
    root: Path = ROOT,
    *,
    scorer: ScoreFunction = score_variant,
) -> tuple[JsonDict, JsonDict, dict[str, dict[str, JsonDict]]]:
    clean = [row for row in variants if row["clean_frozen_selection_eligible"]]
    loco, held_out_predictions = select_loco(variants)
    held_out_scores = scorer(held_out_predictions)
    for fold, case in zip(loco["folds"], CASES):
        for tolerance in TOLERANCE_KEYS:
            replayed_case = next(row for row in held_out_scores[tolerance]["cases"] if row["case_id"] == case)
            if not equivalent(replayed_case, fold["held_out_scores"][tolerance]):
                raise ValueError("Held-out composite replay differs from the fold-selected prediction score.")
    strict = find_variant(variants, STRICT_ID)
    topology_gate = validate_topology_gate(root)
    inventory["evidence_paths"]["topology_checkpoints"] = sorted(
        topology_gate["checkpoint_evidence"]["files"]
    )
    for key, paths in topology_gate["supporting_evidence_paths"].items():
        inventory["evidence_paths"][f"topology_{key}"] = paths
    strict_contract = validate_strict_deployment_contract(strict, inventory, topology_gate)
    strict["origin_policy_certified"] = strict_contract["origin_policy_certified"]
    strict["deployment_eligible"] = strict_contract["deployment_eligible"]
    development_order = sorted(clean, key=lambda row: rank_key(row, CASES))
    development = development_order[0]
    aliases = semantic_aliases(variants)
    retrospective = sorted(
        [row for row in variants if not row["clean_frozen_selection_eligible"]],
        key=lambda row: rank_key(row, CASES),
    )
    heldout_bootstrap = paired_bootstrap(held_out_scores, strict["scores"])
    development_bootstrap = paired_bootstrap(development["scores"], strict["scores"])
    spacing_challenger = find_variant(variants, "deterministic:deterministic-strict-spacing1.5")
    learned_candidates = [
        row for row in clean
        if row["family"] in {"tabular", "cnn"}
        and row["origin_policy_certified"]
        and row["complexity"]["effective_model_dependencies"] > 0
        and (
            row["configuration"].get("profile") == "strict"
            or row["configuration"].get("proposals") == "strict"
        )
    ]
    best_learned_f1 = max(float(subset_summary(row, CASES)["f1"]) for row in learned_candidates)
    tied_learned = sorted(
        row["global_name"] for row in learned_candidates
        if math.isclose(float(subset_summary(row, CASES)["f1"]), best_learned_f1, abs_tol=1e-12)
    )
    challengers = [
        {
            "label": "deterministic-strict-spacing1.5",
            "representative_global_name": spacing_challenger["global_name"],
            "all_five_development_f1_3mm": subset_summary(spacing_challenger, CASES)["f1"],
            "origin_policy_certified": spacing_challenger["origin_policy_certified"],
            "effective_model_dependencies": spacing_challenger["complexity"]["effective_model_dependencies"],
            "symmetric_promotion_gate": False,
            "missing_symmetric_evidence": [
                "same frozen 24-case topology gate", "native organizer Windows/four-core/8-GB timing"
            ],
        },
        {
            "label": "best strict-proposal learned rows",
            "representative_global_name": tied_learned[0],
            "tied_global_names": tied_learned,
            "all_five_development_f1_3mm": best_learned_f1,
            "origin_policy_certified": True,
            "effective_model_dependencies": min(
                row["complexity"]["effective_model_dependencies"]
                for row in learned_candidates
                if row["global_name"] in tied_learned
            ),
            "symmetric_promotion_gate": False,
            "missing_symmetric_evidence": [
                "same frozen 24-case topology gate", "native organizer Windows/four-core/8-GB timing"
            ],
        },
    ]
    rankings = {
        "schema_version": 1,
        "selection_rule": loco["rule"],
        "clean_all_five_development_ranking": [ranked_entry(row, CASES) for row in development_order],
        "retrospective_or_diagnostic_all_five_ranking": [ranked_entry(row, CASES) for row in retrospective],
        "semantic_alias_groups": aliases,
        "interpretation": "All-five ordering is descriptive development evidence, never independent held-out accuracy.",
    }
    report = {
        "schema_version": 1,
        "scope": "Leakage-safe global frozen-family whole-case LOCO selection over five reused development cases.",
        "inventory": {
            **inventory["families"],
            "total_variants": len(variants),
            "clean_frozen_selection_eligible": len(clean),
            "cases": list(CASES),
            "tolerances_mm": list(TOLERANCES),
            "production_family_evidence_validated": not inventory.get("fixture_mode", False),
            "deterministic_replay": inventory.get("deterministic_replay"),
        },
        "selection": {
            **loco,
            "held_out_composite_scores": held_out_scores,
            "signed_count_bias": signed_count_summary(held_out_scores),
            "reference_changes_vs_strict": reference_changes(held_out_scores, strict["scores"]),
            "paired_case_bootstrap_vs_strict": heldout_bootstrap,
        },
        "strict_baseline": {
            "global_name": strict["global_name"],
            "scores": strict["scores"],
            "signed_count_bias": signed_count_summary(strict["scores"]),
            "configuration": strict["configuration"],
            "prediction_provenance": strict["prediction_provenance"],
            "resources": resource_summary(strict),
        },
        "all_five_development_winner": {
            "global_name": development["global_name"],
            "scores": development["scores"],
            "signed_count_bias": signed_count_summary(development["scores"]),
            "configuration": development["configuration"],
            "complexity": development["complexity"],
            "origin_policy_certified": development["origin_policy_certified"],
            "prediction_provenance": development["prediction_provenance"],
            "resources": resource_summary(development),
            "paired_case_bootstrap_vs_strict": development_bootstrap,
            "interpretation": "Descriptive reused-development maximum; not an independent held-out score.",
        },
        "origin_certified_challengers_not_promoted": {
            "rows": challengers,
            "interpretation": "These reused-development challengers outscore strict, but no symmetric promotion gate exists: "
            "they were not all run through the same frozen synthetic cohort and native target hardware evidence is absent.",
        },
        "semantic_aliases": {
            "groups": len(aliases),
            "eligibility_conflicts": sum(row["clean_eligibility_conflict"] for row in aliases),
            "origin_policy_conflicts": sum(row["origin_policy_conflict"] for row in aliases),
            "details_path": "labels/final-eval/selection/rankings.json",
        },
        "deployment_recommendation": {
            "global_name": strict["global_name"],
            "algorithm": "Fixed production deterministic strict detector (no fitted model or per-case switching).",
            "decision": "KEEP STRICT" if strict_contract["deployment_eligible"] else "WITHHOLD",
            "reason": "The LOCO fold composite is not one deployable algorithm. Higher reused-development "
            "origin-certified challengers have no symmetric common synthetic-regression and native target-resource "
            "promotion gate, so development score alone cannot replace the validated incumbent.",
            "origin_policy_certified": strict_contract["origin_policy_certified"],
            "deployment_eligible": strict_contract["deployment_eligible"],
            "strict_contract_validation": strict_contract,
            "synthetic_gate": topology_gate,
            "bundle": "labels/final-eval/selection/predictions/deployment",
            "remaining_gate": "Native organizer Windows/four-core/8-GB timing has not been measured.",
        },
        "resources": {
            "selected_fold_held_out_measurements": [
                {
                    "held_out_case": fold["held_out_case"],
                    "selected_global_name": fold["selected_global_name"],
                    "runtime_s": fold["held_out_runtime_s_not_used_for_selection"],
                }
                for fold in loco["folds"]
            ],
            "comparability": "Upstream runtime scopes and hosts differ; runtime is only a late deterministic tie-breaker. "
            "No native organizer-Windows benchmark exists.",
        },
        "upstream_failures": family_records(reports, "failures"),
        "upstream_exclusions": {
            family: [compact_exclusion(value) for value in report_value.get("exclusions", [])]
            for family, report_value in reports.items()
        },
        "audit_findings": reports["audit"].get("findings", []),
        "limitations": [
            "No official challenge evaluator or official numeric weighted score was supplied.",
            "The 19 judge-approved references retain AI-assisted, non-exhaustive provenance; false positives are local-reference false positives.",
            "Only 3/19 radii are measured; unknown-radius error is masked rather than invented.",
            "Five reused and previously inspected cases cannot establish hidden-test, clinical, or external generalization accuracy.",
            "Whole-case bootstrap intervals quantify small-sample variation but do not remove adaptive selection bias.",
            "Review-union rows may be clean research candidates while remaining uncertified for the final 2-mm origin policy.",
            "Runtime/RSS evidence was collected under differing Linux/macOS scopes, not native organizer Windows hardware.",
        ],
        "provenance": {
            "pinned": inventory["pins"],
            "zip_audit": {
                "source_identifier": "EVAL_SET-20260913T015512Z-1-001.zip",
                "zip_sha256": "7cec39de73c8447864b6103f858768781a201279ea808bd75f153f6fd5eecdbc",
                "manifest_sha256": PINNED_RELEASE_MANIFEST_SHA256,
                "historical_result": "57 safe entries; 55 manifest payloads were recorded byte-identical.",
                "selector_verification": "historical_audit_not_reopened_by_selector",
            },
            "reproduction": [
                "uv venv --python 3.13.3 .venv313",
                "uv pip install --python .venv313/bin/python -r requirements-dev.txt -r requirements-trees.txt -r requirements-cnn-inference.txt -r requirements-resources.txt",
                "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 .venv313/bin/python final_eval_select.py run",
                "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 .venv313/bin/python final_eval_select.py verify",
            ],
        },
    }
    bundles = {"held-out": held_out_predictions, "deployment": strict["predictions"]}
    return report, rankings, bundles


def load_reports(root: Path = ROOT) -> tuple[JsonDict, JsonDict]:
    reports = {}
    provenance = {}
    for family, name in REPORT_PATHS.items():
        path = contained_path(root, name.as_posix())
        reports[family] = strict_read_json(path)
        provenance[family] = {"path": relative(path, root), "sha256": digest(path)}
    return reports, provenance


def write_bundles(output: Path, bundles: Mapping[str, Mapping[str, JsonDict]]) -> None:
    for bundle_name, predictions in bundles.items():
        if set(predictions) != set(CASES):
            raise ValueError(f"{bundle_name} bundle is incomplete.")
        directory = output / "predictions" / bundle_name
        for case in CASES:
            write_json(directory / f"{case}.json", predictions[case])
        if {path.name for path in directory.glob("*.json")} != {f"{case}.json" for case in CASES}:
            raise ValueError(f"{bundle_name} output contains stale prediction JSON files.")


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_results(report: JsonDict) -> str:
    selection = report["selection"]
    held = selection["held_out_composite_scores"]
    strict = report["strict_baseline"]["scores"]
    development = report["all_five_development_winner"]
    lines = [
        "# Final evaluation results",
        "",
        "## Recommendation",
        "",
        "**Keep the fixed production `deterministic-strict` algorithm.** Its deployment eligibility and origin-policy "
        "certification are derived from an exact full production configuration, pinned frozen specification, raw "
        "deterministic receipts, and the replayed topology gate—not from its name. The global LOCO result below "
        "estimates a selection procedure whose five predictions can come from different algorithms; it is not a "
        "single deployable configuration. The all-five winner is development-only.",
        "",
        "No official evaluator was supplied, so no official weighted challenge score is reported. These are local "
        "maximum-cardinality one-to-one ostium matches against 19 judge-approved but AI-assisted/non-exhaustive targets.",
        "",
        "## Leakage-safe global leave-one-case-out selection",
        "",
        f"Validated {report['inventory']['total_variants']} variants from all five required reports; "
        f"{report['inventory']['clean_frozen_selection_eligible']} were eligible clean frozen candidates. Each fold was "
        "ranked using only the other four cases: pooled 3-mm F1, count MAE, effective model dependencies, algorithm "
        "components, measured runtime, then global lexical name.",
        "",
        "| Held out | Four-case selected variant | Dev F1 | Dev count MAE | Dependencies | Dev runtime (s) | Held-out TP/FP/FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in selection["folds"]:
        rank = fold["selection_rank_key"]
        score = fold["held_out_scores"]["3"]
        lines.append(
            f"| {fold['held_out_case']} | `{fold['selected_global_name']}` | "
            f"{format_number(rank['pooled_f1_3mm'])} | {format_number(rank['count_mae_3mm'])} | "
            f"{rank['effective_model_dependencies']} | {format_number(rank['measured_runtime_s'], 3)} | "
            f"{score['true_positives']}/{score['false_positives']}/{score['false_negatives']} |"
        )
    lines.extend([
        "",
        "Selection stability: " + ", ".join(
            f"`{name}` {count}/5" for name, count in selection["selection_stability"].items()
        ) + ".",
        "",
        "| Tolerance | Composite TP/FP/FN | Precision | Recall | F1 | Count MAE | Mean signed count bias | Strict F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for tolerance in TOLERANCE_KEYS:
        summary = held[tolerance]["summary"]
        baseline = strict[tolerance]["summary"]
        signed = selection["signed_count_bias"][tolerance]["mean"]
        lines.append(
            f"| {tolerance} mm | {summary['true_positives']}/{summary['false_positives']}/{summary['false_negatives']} | "
            f"{format_number(summary['precision'])} | {format_number(summary['recall'])} | "
            f"{format_number(summary['f1'])} | {format_number(summary['count_mae'])} | "
            f"{format_number(signed)} | {format_number(baseline['f1'])} |"
        )
    bootstrap = selection["paired_case_bootstrap_vs_strict"]["tolerances"]["3"]
    lines.extend([
        "",
        "At 3 mm, the fixed-decision paired whole-case bootstrap (10,000 resamples, seed 20260913) gives a "
        f"composite-minus-strict F1 interval of [{format_number(bootstrap['paired_f1_delta_95_percentile'][0])}, "
        f"{format_number(bootstrap['paired_f1_delta_95_percentile'][1])}]. With only five reused cases this is "
        "descriptive uncertainty, not a correction for selection bias.",
        "",
        "Exact gained/lost reference IDs versus strict at 2/3/5 mm are preserved in "
        "`labels/final-eval/selection/report.json`.",
        "",
        "## All-five development winner (not held out)",
        "",
        f"`{development['global_name']}` is the descriptive all-five maximum. It is explicitly **not independent "
        "held-out accuracy** and is not the deployment recommendation.",
        "",
        "| Tolerance | TP/FP/FN | Precision | Recall | F1 | Count MAE | Origin-policy certified |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for tolerance in TOLERANCE_KEYS:
        summary = development["scores"][tolerance]["summary"]
        lines.append(
            f"| {tolerance} mm | {summary['true_positives']}/{summary['false_positives']}/{summary['false_negatives']} | "
            f"{format_number(summary['precision'])} | {format_number(summary['recall'])} | "
            f"{format_number(summary['f1'])} | {format_number(summary['count_mae'])} | "
            f"{str(development['origin_policy_certified']).lower()} |"
        )
    lines.extend([
        "",
        "Its exact threshold/weight, detector settings, model/source hashes, prediction hashes, and runtime are in the "
        "machine-readable report. The complete clean and retrospective rankings are in `rankings.json`.",
        "",
        "## Origin-certified challengers without symmetric promotion evidence",
        "",
        "Some origin-certified candidates score higher than strict on the same five reused development cases. They "
        "remain unpromoted because they lack a symmetric promotion gate, not because their descriptive score is lower:",
        "",
        "| Challenger | All-five 3 mm F1 | Origin certified | Model dependencies | Symmetric topology/native-resource gate |",
        "| --- | ---: | --- | ---: | --- |",
    ])
    for row in report["origin_certified_challengers_not_promoted"]["rows"]:
        label = row["label"]
        if row.get("tied_global_names"):
            label += f" ({len(row['tied_global_names'])} tied rows; e.g. `{row['representative_global_name']}`)"
        else:
            label = f"`{row['representative_global_name']}`"
        lines.append(
            f"| {label} | {format_number(row['all_five_development_f1_3mm'])} | "
            f"{str(row['origin_policy_certified']).lower()} | {row['effective_model_dependencies']} | no |"
        )
    lines.extend([
        "",
        "`deterministic-strict-spacing1.5` is approximately F1 0.5946; the best strict-proposal learned rows are "
        "approximately F1 0.5926. Neither class was exercised through the exact same frozen 24-case topology gate "
        "and native organizer hardware measurement as a promotion candidate, so no symmetric replacement decision "
        "is supported.",
        "",
        "## Synthetic topology and deployment gate",
        "",
        "The selector enumerated and semantically replayed the exact frozen 11×24 checkpoint tree (264/264), "
        "including case/config/root-separation/runtime identity, 2/3/5-mm scores, aggregate/runtime maps, and zero "
        "failures. At 3 mm:",
        "",
        "| Variant | TP/FP/FN | F1 | Count MAE |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in report["deployment_recommendation"]["synthetic_gate"]["variants"]:
        summary = row["scores_3mm"]
        lines.append(
            f"| `{row['name']}` | {summary['true_positives']}/{summary['false_positives']}/{summary['false_negatives']} | "
            f"{format_number(summary['f1'])} | {format_number(summary['count_mae'])} |"
        )
    lines.extend([
        "",
        "Strict had 46/0/3 (F1 0.9684) and zero negative-control detections. Although review-origin2 reached "
        "47/0/2, topology recovery variants were post-reference diagnostics; several added substantial synthetic "
        "false positives. None is promoted. Final learned candidates were not all exercised through this identical "
        "gate, so score gains alone cannot certify deployment.",
        "",
        "The deployment bundle contains five predictions from the single fixed strict algorithm at "
        "`labels/final-eval/selection/predictions/deployment/`. The held-out fold composite is separately stored under "
        "`predictions/held-out/` and must not be submitted as one algorithm.",
        "",
        "## Resources, exclusions, and limitations",
        "",
        "| Configuration | Mean runtime/case (s) | Runtime range (s) | Maximum peak RSS (MiB) |",
        "| --- | ---: | ---: | ---: |",
        f"| Fixed strict | {format_number(report['strict_baseline']['resources']['runtime_s']['mean'], 3)} | "
        f"{format_number(report['strict_baseline']['resources']['runtime_s']['range'][0], 3)}–"
        f"{format_number(report['strict_baseline']['resources']['runtime_s']['range'][1], 3)} | "
        f"{format_number(report['strict_baseline']['resources']['maximum_peak_rss_mb'], 2)} |",
        f"| Development-only winner | {format_number(development['resources']['runtime_s']['mean'], 3)} | "
        f"{format_number(development['resources']['runtime_s']['range'][0], 3)}–"
        f"{format_number(development['resources']['runtime_s']['range'][1], 3)} | "
        f"{format_number(development['resources']['maximum_peak_rss_mb'], 2)} |",
        "",
        "Strict also outperformed the adaptive held-out composite at 3 mm: 8/3/11, F1 0.5333 and count "
        "MAE 2.0 versus 8/9/11, F1 0.4444 and count MAE 3.2. The all-five development winner's higher "
        "F1 0.6207 is retrospective and uses the relaxed review-union origin policy.",
        "",
        "Runtime scopes differ by family (deterministic source inference/audit, tabular filtering/model scope, CNN cold "
        "worker extraction, and topology stage timing), so runtime is only a late tie-breaker. Linux/macOS measurements "
        "do not establish native organizer Windows/four-core/8-GB performance.",
        "",
        "All upstream failure lists were empty at aggregation. Incompatible old tree/blend artifacts, historical models "
        "with released-case overlap or source-contract mismatch, origin-disabled rows, trace ceilings, all post-reference "
        "topology rows, and the audit duplicate were excluded from clean selection. Review-union rows remain research "
        "candidates where their frozen report permits them, but are not final-origin-policy certified.",
        "",
        "Only 3 of 19 reference radii are measured; the other 16 remain unknown and are masked from radius error. "
        "Unmatched predictions are false positives only relative to the non-exhaustive local reference set. Five "
        "previously inspected cases cannot support hidden-test, clinical, or external-generalization claims.",
        "",
        "## Reproduction and provenance",
        "",
        "Run `.venv313/bin/python final_eval_select.py verify` under the four-thread environment to replay all 1,400 "
        "source prediction files, all 130 deterministic receipts and 260 deterministic outputs, and all 264 topology "
        "checkpoints; it also validates family lineage, recomputes every 2/3/5-mm score, and verifies the complete "
        "evidence manifest. Canonical pinned Git LFS pointers are accepted for selector-only identity replay and are "
        "reported as unresolved; inference still requires payload bytes. The evaluated scorer is pinned to "
        f"`{PINNED_SCORER_SHA256}` and the production detector to `{PINNED_DETECTOR_SHA256}`.",
        "",
        "The historical audited download source identifier is `EVAL_SET-20260913T015512Z-1-001.zip`, SHA-256 "
        "`7cec39de73c8447864b6103f858768781a201279ea808bd75f153f6fd5eecdbc`. The selector does not reopen that "
        "archive; it validates the portable basename/digest record and repository evidence. Exact per-file evidence "
        "hashes and tree-root digests are in `labels/final-eval/selection/manifest.json`.",
        "",
    ])
    return "\n".join(lines)


def evidence_tree(
    root: Path, paths: Sequence[str], *, expected_count: int | None = None, context: str
) -> JsonDict:
    names = sorted(set(paths))
    if len(names) != len(paths):
        raise ValueError(f"Duplicate paths in {context} evidence inventory.")
    if expected_count is not None and len(names) != expected_count:
        raise ValueError(f"{context} evidence count drifted: expected {expected_count}, got {len(names)}.")
    files = {}
    for name in names:
        path = contained_path(root, name)
        files[name] = digest(path)
    if any(name == "labels/final-eval/selection/manifest.json" for name in files):
        raise ValueError("Selection manifest must not hash itself.")
    return {"count": len(files), "root_sha256": canonical_sha256(files), "files": files}


def build_manifest(
    root: Path,
    output: Path,
    report_provenance: JsonDict,
    report: JsonDict,
    bundles: Mapping[str, Mapping[str, JsonDict]],
    inventory: JsonDict,
) -> JsonDict:
    evidence_paths = inventory["evidence_paths"]
    groups = {
        "deterministic_receipts": evidence_tree(
            root, evidence_paths["deterministic_receipts"], expected_count=130,
            context="deterministic receipts",
        ),
        "deterministic_predictions": evidence_tree(
            root, evidence_paths["deterministic_predictions"], expected_count=260,
            context="deterministic predictions",
        ),
        "topology_checkpoints": evidence_tree(
            root, evidence_paths["topology_checkpoints"], expected_count=264,
            context="topology checkpoints",
        ),
        "deterministic_supporting": evidence_tree(
            root, list(DETERMINISTIC_SUPPORTING_PATHS), expected_count=7,
            context="deterministic supporting artifacts",
        ),
        "topology_supporting": evidence_tree(
            root, list(TOPOLOGY_SUPPORTING_PATHS), expected_count=5,
            context="topology supporting artifacts",
        ),
        "topology_real_predictions": evidence_tree(
            root, evidence_paths["topology_real_predictions"], expected_count=55,
            context="topology real predictions",
        ),
        "topology_diagnoses": evidence_tree(
            root, evidence_paths["topology_diagnoses"], expected_count=55,
            context="topology diagnoses",
        ),
        "topology_resources": evidence_tree(
            root, evidence_paths["topology_resources"], expected_count=45,
            context="topology resources",
        ),
        "selector_protocol_resource_wrapper": evidence_tree(
            root, list(SELECTOR_SUPPORTING_PATHS), expected_count=3,
            context="selector/protocol/resource-wrapper",
        ),
        "release_identity": evidence_tree(
            root, ["labels/organizer-v1/validation.json", "eval/docs/manifest.json"],
            expected_count=2, context="release identity",
        ),
        "tabular_lineage": evidence_tree(root, evidence_paths["tabular"], context="tabular lineage"),
        "cnn_lineage": evidence_tree(root, evidence_paths["cnn"], context="CNN lineage"),
        "tabular_predictions": evidence_tree(
            root, evidence_paths["tabular_predictions"], expected_count=530,
            context="tabular predictions",
        ),
        "cnn_predictions": evidence_tree(
            root, evidence_paths["cnn_predictions"], expected_count=560,
            context="CNN predictions",
        ),
        "audit_predictions": evidence_tree(
            root, evidence_paths["audit_predictions"], expected_count=5,
            context="audit predictions",
        ),
    }
    results_path = root / "FINAL_EVALUATION_RESULTS.md"
    outputs = {}
    for path in (output / "report.json", output / "rankings.json", results_path):
        outputs[relative(path, root)] = digest(path)
    bundle_rows: dict[str, dict[str, JsonDict]] = {}
    for bundle_name, predictions in bundles.items():
        bundle_rows[bundle_name] = {}
        for case in CASES:
            path = output / "predictions" / bundle_name / f"{case}.json"
            if strict_read_json(path) != predictions[case]:
                raise ValueError(f"Written {bundle_name} prediction differs for {case}.")
            bundle_rows[bundle_name][case] = {
                "path": relative(path, root),
                "sha256": digest(path),
                "semantic_sha256": canonical_sha256(predictions[case]),
            }
    return {
        "schema_version": 2,
        "cases": list(CASES),
        "tolerances_mm": list(TOLERANCES),
        "selection_rule": report["selection"]["rule"],
        "bootstrap": {"unit": "whole case", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "inputs": {"reports": report_provenance, "evidence_trees": groups},
        "pinned": report["provenance"]["pinned"],
        "selected_source_predictions": {
            fold["held_out_case"]: fold["held_out_source_prediction"] for fold in report["selection"]["folds"]
        },
        "deployment_source_predictions": report["strict_baseline"]["prediction_provenance"]["byte_sha256"],
        "outputs": outputs,
        "prediction_bundles": bundle_rows,
        "self_hash_included": False,
        "official_evaluator_supplied": False,
        "native_windows_target_measured": False,
    }


def evaluate(
    root: Path = ROOT,
) -> tuple[JsonDict, JsonDict, dict[str, dict[str, JsonDict]], JsonDict, JsonDict]:
    reports, report_provenance = load_reports(root)
    variants, inventory = validate_inventory(reports, root)
    report, rankings, bundles = build_evaluation(reports, variants, inventory, root)
    return report, rankings, bundles, report_provenance, inventory


def run(root: Path = ROOT, output: Path = OUTPUT) -> JsonDict:
    report, rankings, bundles, report_provenance, inventory = evaluate(root)
    write_bundles(output, bundles)
    write_json(output / "report.json", report)
    write_json(output / "rankings.json", rankings)
    (root / "FINAL_EVALUATION_RESULTS.md").write_text(render_results(report), encoding="utf-8")
    manifest = build_manifest(root, output, report_provenance, report, bundles, inventory)
    write_json(output / "manifest.json", manifest)
    return report


def verify(root: Path = ROOT, output: Path = OUTPUT) -> JsonDict:
    report, rankings, bundles, report_provenance, inventory = evaluate(root)
    if strict_read_json(output / "report.json") != report:
        raise ValueError("Selection report is stale or differs from replay.")
    if strict_read_json(output / "rankings.json") != rankings:
        raise ValueError("Selection rankings are stale or differ from replay.")
    if (root / "FINAL_EVALUATION_RESULTS.md").read_text(encoding="utf-8") != render_results(report):
        raise ValueError("Human-readable final results are stale.")
    expected_manifest = build_manifest(root, output, report_provenance, report, bundles, inventory)
    if strict_read_json(output / "manifest.json") != expected_manifest:
        raise ValueError("Selection manifest is stale or contains a hash mismatch.")
    for bundle_name, predictions in bundles.items():
        for case, expected in predictions.items():
            path = output / "predictions" / bundle_name / f"{case}.json"
            if strict_read_json(path) != expected:
                raise ValueError(f"{bundle_name} output drift for {case}.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    result = run() if args.command == "run" else verify()
    score = result["selection"]["held_out_composite_scores"]["3"]["summary"]
    print(json.dumps({
        "validated_variants": result["inventory"]["total_variants"],
        "eligible_variants": result["inventory"]["clean_frozen_selection_eligible"],
        "held_out_3mm": {key: score[key] for key in (
            "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "count_mae"
        )},
        "development_winner": result["all_five_development_winner"]["global_name"],
        "deployment": result["deployment_recommendation"]["global_name"],
    }, indent=2))


if __name__ == "__main__":
    main()
