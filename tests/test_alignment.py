import pytest

from src.reasoning_compression import alignment


def test_get_original_rows_by_position_streams_requested_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"response": "row 0"},
        {"response": "row 1"},
        {"response": "row 2"},
    ]

    def fake_load_dataset(
        dataset_id: str,
        split: str,
        streaming: bool,
    ) -> list[dict[str, str]]:
        assert dataset_id == "dataset"
        assert split == "train"
        assert streaming is True
        return rows

    monkeypatch.setattr(alignment, "load_dataset", fake_load_dataset)

    original_rows = alignment.get_original_rows_by_position(
        [0, 2],
        dataset_id="dataset",
        split="train",
    )

    assert original_rows == {
        0: {"response": "row 0"},
        2: {"response": "row 2"},
    }


def test_get_original_rows_by_position_validates_trace_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        alignment.get_original_rows_by_position([])

    with pytest.raises(ValueError, match="non-negative"):
        alignment.get_original_rows_by_position([-1])
