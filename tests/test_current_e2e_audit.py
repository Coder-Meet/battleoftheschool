import current_e2e_audit as audit


def test_duplicate_profile_representations_do_not_inflate_reference_coverage(monkeypatch):
    monkeypatch.setattr(audit, "CASES", ("case",))
    monkeypatch.setattr(audit, "load_reference", lambda case: {
        "daughters": [{"instance_id": "first"}, {"instance_id": "second"}],
    })
    block = {tolerance: {"cases": [{"matches": [{"reference_id": "first"}]}]}
             for tolerance in ("2", "3", "5")}
    result = audit.proposal_coverage({"strict/plain": block, "review/plain": block})
    for value in result["by_tolerance"].values():
        assert value["matched_targets"] == 1
        assert value["cases"][0]["unmatched_reference_ids"] == ["second"]
