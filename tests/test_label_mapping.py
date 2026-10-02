import pandas as pd
import pytest

from src import config
from src.classifier.data_acquisition import apply_mapping, clean_duplicates, validate_mapping_names
from src.classifier.label_mapping import BANKING77_TO_INTENT, CLINC150_TO_INTENT


def test_mapping_targets_are_valid_intents():
    for mapping in (BANKING77_TO_INTENT, CLINC150_TO_INTENT):
        assert set(mapping.values()) <= set(config.INTENTS)


def test_apply_mapping_keeps_only_mapped_labels_case_insensitive():
    rows = [("refund please", "Request_Refund"), ("lost my card", "lost_or_stolen_card")]
    df = apply_mapping(rows, {"request_refund": "billing"}, "fake")
    assert len(df) == 1
    assert df.iloc[0]["label"] == "billing"
    assert df.iloc[0]["original_label"] == "Request_Refund"


def test_apply_mapping_rejects_unknown_intent():
    with pytest.raises(ValueError):
        apply_mapping([("x", "a")], {"a": "not_an_intent"}, "fake")


def test_validate_mapping_names_catches_typos():
    with pytest.raises(ValueError):
        validate_mapping_names({"requst_refund": "billing"}, ["request_refund"], "fake")
    validate_mapping_names({"Request_Refund": "billing"}, ["request_refund"], "fake")  # ok


def test_clean_duplicates_removes_dupes_and_conflicts():
    df = pd.DataFrame({
        "text": ["hello", "Hello ", "pay bill", "pay bill"],
        "label": ["general", "general", "billing", "technical"],
        "source": ["a"] * 4, "original_label": ["x"] * 4,
    })
    out = clean_duplicates(df)
    assert list(out["text"]) == ["hello"]   # dupe collapsed, conflicting 'pay bill' dropped
