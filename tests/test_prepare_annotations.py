import csv
import json

import numpy as np
import pytest
import SimpleITK as sitk

from prepare_annotations import SCOPE, point_index, prepare, proposal_packet, render_packet, write_worksheet
from score_references import load_references, normalize_reference


def image_and_prediction():
    image = sitk.Image([12, 14, 16], sitk.sitkInt16)
    image.SetSpacing((0.7, 0.8, 1.5))
    image.SetOrigin((-20, 30, 100))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    branch = {
        "instance_id": "branch_001", "parent_instance_id": "aorta",
        "ostium_xyz_mm": list(image.TransformContinuousIndexToPhysicalPoint([5.5, 6, 7])),
        "seed_xyz_mm": list(image.TransformContinuousIndexToPhysicalPoint([5.5, 6, 10])),
        "radius_mm": 1, "direction_xyz": [0, 0, 1],
    }
    return image, {"case_id": "subject001", "parent": {"instance_id": "aorta"}, "daughters": [branch]}


def test_rotated_geometry_and_provenance_do_not_confirm_model_agreement(tmp_path):
    image, prediction = image_and_prediction()
    packet = proposal_packet("subject001", image, {"production": prediction, "experiment": prediction})
    assert packet["scope"] == SCOPE
    assert packet["case_review"]["eligible_daughter_count"] is None
    assert packet["case_review"]["entire_parent_reviewed"] is False
    assert "daughters" not in packet
    assert len(packet["candidates"]) == 1
    candidate = packet["candidates"][0]
    assert len(candidate["sources"]) == 2
    assert np.allclose(candidate["ostium_index_xyz"], [5.5, 6, 7])
    assert candidate["review"]["status"] == "unreviewed"
    assert candidate["review"]["corrected_ostium_xyz_mm"] is None
    worksheet = tmp_path / "review.csv"
    write_worksheet(packet, worksheet)
    with worksheet.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "unreviewed"
    assert row["measured_radius_mm"] == ""


def test_nearby_ostia_and_different_measurements_are_not_automatically_merged():
    image, prediction = image_and_prediction()
    second = json.loads(json.dumps(prediction))
    second["daughters"][0]["ostium_xyz_mm"][0] += 0.8
    packet = proposal_packet("subject001", image, {"production": prediction, "experiment": second})
    assert len(packet["candidates"]) == 2
    assert packet["candidates"][0]["nearby_proposals_within_3mm"] == ["candidate_002"]
    second["daughters"][0]["ostium_xyz_mm"] = prediction["daughters"][0]["ostium_xyz_mm"]
    second["daughters"][0]["radius_mm"] = 2
    assert len(proposal_packet("subject001", image, {"a": prediction, "b": second})["candidates"]) == 2


def test_zero_proposals_are_not_a_verified_negative_and_scorer_refuses_packet(tmp_path):
    image, prediction = image_and_prediction()
    prediction["daughters"] = []
    packet = proposal_packet("subject001", image, {"production": prediction})
    assert packet["candidates"] == []
    assert packet["case_review"]["eligible_daughter_count"] is None
    with pytest.raises(ValueError, match="Unreviewed"):
        normalize_reference(packet, [])
    path = tmp_path / "proposals.json"
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="Unreviewed"):
        load_references(path)
    with pytest.raises(ValueError, match="Unreviewed"):
        normalize_reference({"scope": SCOPE, **prediction}, [])


def test_outside_coordinates_case_mismatch_and_review_overwrite_are_rejected(tmp_path):
    image, prediction = image_and_prediction()
    with pytest.raises(ValueError, match="outside"):
        point_index(image, [1000.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="case ID"):
        proposal_packet("subject002", image, {"production": prediction})
    with pytest.raises(FileExistsError, match="protect"):
        prepare(tmp_path, [], [], tmp_path)


def test_native_pdf_survey_covers_every_slice_and_checks_mask_geometry(tmp_path):
    image, prediction = image_and_prediction()
    array = np.zeros((16, 14, 12), dtype=np.uint8)
    array[2:14, 4:10, 3:9] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)
    packet = proposal_packet("subject001", image, {"production": prediction})
    pages = render_packet(image, mask, packet, tmp_path)
    assert pages["survey_k_indices"] == list(range(16))
    assert pages["proposal_pages"] == 1
    for name in ("01_blinded_survey.pdf", "02_proposed_openings.pdf"):
        assert (tmp_path / name).read_bytes().startswith(b"%PDF")
    mask.SetOrigin((0, 0, 0))
    with pytest.raises(ValueError, match="geometry differ"):
        render_packet(image, mask, packet, tmp_path)
