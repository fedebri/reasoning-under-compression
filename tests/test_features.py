from pathlib import Path

import numpy as np
import pytest

from src.reasoning_compression.features import (
    NOT_APPLICABLE_DIFFICULTY,
    build_feature_chunk,
    latest_feature_build_dir,
    normalize_difficulty,
    parse_response,
)


def test_parse_response_extracts_blocks_summaries_and_answer() -> None:
    response = (
        "<think>"
        "<|block_start|>First reasoning block<|block_end|>"
        "<|summary_start|>First summary<|summary_end|>"
        "<|block_start|>Second reasoning block<|block_end|>"
        "<|summary_start|>Second summary<|summary_end|>"
        "</think>\nFinal answer"
    )

    parsed = parse_response(response)

    assert parsed["blocks"] == ["First reasoning block", "Second reasoning block"]
    assert parsed["summaries"] == ["First summary", "Second summary"]
    assert parsed["answer_text"] == "Final answer"
    assert parsed["n_blocks"] == 2
    assert parsed["n_summaries"] == 2


def test_parse_response_treats_missing_think_as_answer_only() -> None:
    parsed = parse_response("Final answer only")

    assert parsed["think_text"] == ""
    assert parsed["answer_text"] == "Final answer only"
    assert parsed["blocks"] == []
    assert parsed["summaries"] == []


@pytest.mark.parametrize("difficulty", [None, "", "   ", np.nan])
def test_normalize_difficulty_uses_explicit_not_applicable(
    difficulty: object,
) -> None:
    assert normalize_difficulty(difficulty) == NOT_APPLICABLE_DIFFICULTY


def test_normalize_difficulty_stringifies_present_values() -> None:
    assert normalize_difficulty(7) == "7"
    assert normalize_difficulty("hard") == "hard"


def test_build_feature_chunk_normalizes_missing_difficulty() -> None:
    rows = [
        {
            "problem": "A) 1 B) 2?",
            "response": (
                "<think>"
                "<|block_start|>Compute 1 + 1.<|block_end|>"
                "<|summary_start|>Adds numbers.<|summary_end|>"
                "</think> 2"
            ),
            "domain": "math",
            "source": "synthetic",
            "difficulty": None,
        }
    ]

    df_traces, df_blocks = build_feature_chunk(rows, start_trace_id=10)

    assert df_traces.loc[0, "trace_id"] == 10
    assert df_traces.loc[0, "difficulty"] == NOT_APPLICABLE_DIFFICULTY
    assert df_blocks.loc[0, "difficulty"] == NOT_APPLICABLE_DIFFICULTY
    assert df_blocks.loc[0, "relative_block_position"] == 0
    assert df_blocks.loc[0, "summary_to_block_token_ratio"] > 0
    assert df_traces.loc[0, "problem_has_multiple_choice"] == 1


def test_build_feature_chunk_validates_required_row_fields() -> None:
    rows = [
        {
            "problem": "Problem text",
            "domain": "math",
            "source": "synthetic",
        }
    ]

    with pytest.raises(ValueError, match="response"):
        build_feature_chunk(rows, start_trace_id=0)


def test_build_feature_chunk_rejects_non_text_problem() -> None:
    rows = [
        {
            "problem": 123,
            "response": "answer",
            "domain": "math",
            "source": "synthetic",
            "difficulty": None,
        }
    ]

    with pytest.raises(TypeError, match="problem"):
        build_feature_chunk(rows, start_trace_id=0)


def test_latest_feature_build_dir_returns_newest_complete_build(
    tmp_path: Path,
) -> None:
    old_build = tmp_path / "20240101_000000"
    new_build = tmp_path / "20240201_000000"
    incomplete_build = tmp_path / "20240301_000000"

    old_build.mkdir()
    new_build.mkdir()
    incomplete_build.mkdir()

    for build_dir in (old_build, new_build):
        (build_dir / "blocks_features_full_labeled.parquet").touch()
        (build_dir / "traces_features_full_labeled.parquet").touch()
    (incomplete_build / "blocks_features_full_labeled.parquet").touch()

    assert latest_feature_build_dir(tmp_path) == new_build
