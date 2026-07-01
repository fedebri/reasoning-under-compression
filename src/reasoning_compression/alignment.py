"""Alignment diagnostics for local feature tables and source rows."""

from collections.abc import Mapping
from itertools import islice

from datasets import load_dataset

from src.reasoning_compression.features import DATASET_ID, SPLIT

__all__ = ["get_original_rows_by_position"]


def get_original_rows_by_position(
    trace_ids: list[int],
    dataset_id: str = DATASET_ID,
    split: str = SPLIT,
) -> dict[int, Mapping[str, object]]:
    """Return streamed source rows whose positions match selected trace IDs.

    Full source-row alignment is intentionally opt-in because it streams the
    Hugging Face dataset. The engineered feature pipeline assigns ``trace_id``
    from streaming order, so source row position ``i`` should correspond to
    ``trace_id == i``.

    Args:
        trace_ids: Non-empty list of non-negative trace IDs to retrieve.
        dataset_id: Hugging Face dataset identifier.
        split: Dataset split to stream.

    Returns:
        Mapping from requested trace ID to the corresponding source row.
    """
    if not trace_ids:
        raise ValueError("trace_ids must contain at least one trace ID.")
    if any(trace_id < 0 for trace_id in trace_ids):
        raise ValueError("trace_ids must be non-negative.")

    target_ids = set(trace_ids)
    max_id = max(target_ids)

    ds_stream = load_dataset(dataset_id, split=split, streaming=True)
    original_rows: dict[int, Mapping[str, object]] = {}

    for row_index, row in enumerate(islice(ds_stream, max_id + 1)):
        if row_index in target_ids:
            original_rows[row_index] = row

        if len(original_rows) == len(target_ids):
            break

    missing_trace_ids = target_ids.difference(original_rows)
    if missing_trace_ids:
        raise ValueError(
            "Could not retrieve original rows for trace_ids: "
            f"{sorted(missing_trace_ids)}"
        )

    return original_rows
