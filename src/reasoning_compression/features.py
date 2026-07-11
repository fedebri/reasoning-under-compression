"""Feature engineering utilities for reasoning compression analysis.

This module contains reusable helpers for parsing OpenMementos responses,
counting tokens, extracting problem-level features, building trace-level and
block-level feature tables, and formatting notebook outputs.

The main data representation has two levels:

- trace-level rows: one row per original problem-response pair
- block-level rows: one row per aligned reasoning block and summary pair

Generated parquet files are intended to be stored locally under the ignored
``data/`` directory, not committed to Git.
"""

from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
import re
from typing import TypedDict

import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm


DATASET_ID = "microsoft/OpenMementos"
SPLIT = "train"
ENCODING_NAME = "cl100k_base"
NOT_APPLICABLE_DIFFICULTY = "not_applicable"

BLOCK_RE = re.compile(r"<\|block_start\|>(.*?)<\|block_end\|>", re.DOTALL)
SUMMARY_RE = re.compile(r"<\|summary_start\|>(.*?)<\|summary_end\|>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
MATH_SYMBOLS = set("+-=*/^<>≤≥≈≠√∑∫π%()[]{}")
REQUIRED_ROW_FIELDS = ("problem", "response", "domain", "source")
FEATURE_BUILD_FILES = (
    "blocks_features_full_labeled.parquet",
    "traces_features_full_labeled.parquet",
)

encoding = tiktoken.get_encoding(ENCODING_NAME)


class ParsedResponse(TypedDict):
    """Structured representation extracted from an OpenMementos response."""

    think_text: str
    answer_text: str
    blocks: list[str]
    summaries: list[str]
    n_blocks: int
    n_summaries: int
    think_chars: int
    answer_chars: int
    response_chars: int


class ProblemFeatures(TypedDict):
    """Problem-level predictor features."""

    problem_chars: int
    problem_tokens: int
    problem_math_symbol_share: float
    problem_has_multiple_choice: int
    problem_has_code_fence: int
    problem_question_mark_count: int


class SpacyProblemFeatures(TypedDict):
    """Optional spaCy-derived problem features."""

    spacy_sentence_count: int
    spacy_avg_sentence_tokens: float
    spacy_stopword_share: float
    spacy_punctuation_share: float
    spacy_numeric_token_share: float
    spacy_oov_share: float
    spacy_noun_share: float
    spacy_verb_share: float
    spacy_entity_count: int


class BuildSummary(TypedDict):
    """Summary returned after writing feature partitions."""

    output_dir: str
    n_trace_rows: int
    n_block_rows: int
    n_parts: int


__all__ = [
    "DATASET_ID",
    "SPLIT",
    "ENCODING_NAME",
    "NOT_APPLICABLE_DIFFICULTY",
    "count_tokens",
    "count_tokens_batch",
    "normalize_difficulty",
    "parse_response",
    "extract_problem_features",
    "build_feature_tables",
    "extract_spacy_problem_features",
    "build_feature_chunk",
    "write_feature_partitions",
    "latest_feature_build_dir",
    "count_share_table",
    "format_describe",
]


def _row_label(row_index: int | None) -> str:
    """Return a compact label for validation error messages."""

    return f"row {row_index}" if row_index is not None else "row"


def _coerce_optional_text(
    value: object | None,
    field_name: str,
    row_index: int | None = None,
) -> str:
    """Return text from external data, treating ``None`` as empty text."""

    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(
            f"Expected {_row_label(row_index)} field '{field_name}' to be "
            f"str or None, received {type(value).__name__}."
        )
    return value


def _coerce_optional_metadata(
    value: object | None,
    field_name: str,
    row_index: int | None = None,
) -> str | None:
    """Validate optional text metadata from external rows."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"Expected {_row_label(row_index)} field '{field_name}' to be "
            f"str or None, received {type(value).__name__}."
        )
    return value


def normalize_difficulty(value: object | None) -> str:
    """Normalize missing difficulty metadata to an explicit category.

    OpenMementos provides task difficulty mainly for coding problems. For
    non-code domains, keeping missing values as ``NaN`` causes modeling frames
    that call ``dropna`` to silently discard whole domains. This helper maps
    missing difficulty to ``"not_applicable"`` and stringifies present scalar
    values so categorical encoders receive one consistent dtype.
    """

    if value is None:
        return NOT_APPLICABLE_DIFFICULTY
    if isinstance(value, str):
        difficulty = value.strip()
        return difficulty if difficulty else NOT_APPLICABLE_DIFFICULTY
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        raise TypeError(
            "Expected difficulty to be a scalar value or None, received "
            f"{type(value).__name__}."
        )
    if bool(pd.isna(value)):
        return NOT_APPLICABLE_DIFFICULTY
    return str(value)


def _validate_dataset_row(
    row: Mapping[str, object],
    row_index: int | None = None,
) -> None:
    """Validate the external row schema needed by feature construction."""

    missing_fields = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ValueError(
            f"Missing required field(s) in {_row_label(row_index)}: {fields}."
        )

    _coerce_optional_text(row.get("problem"), "problem", row_index)
    _coerce_optional_text(row.get("response"), "response", row_index)
    _coerce_optional_metadata(row.get("domain"), "domain", row_index)
    _coerce_optional_metadata(row.get("source"), "source", row_index)
    normalize_difficulty(row.get("difficulty"))


def _metadata_from_row(
    row: Mapping[str, object],
    row_index: int | None = None,
) -> dict[str, str | None]:
    """Extract validated metadata from a dataset row."""

    return {
        "domain": _coerce_optional_metadata(row.get("domain"), "domain", row_index),
        "source": _coerce_optional_metadata(row.get("source"), "source", row_index),
        "difficulty": normalize_difficulty(row.get("difficulty")),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator`` or ``np.nan`` for zero denominator."""

    return numerator / denominator if denominator else np.nan


def _safe_savings(compressed_length: int, original_length: int) -> float:
    """Return proportional compression savings for positive original length."""

    return 1 - (compressed_length / original_length) if original_length else np.nan


def count_tokens(text: str | None) -> int:
    """Count tokens in a text string using the configured tiktoken encoding.

    Args:
        text: Input text. ``None`` is treated as an empty string.

    Returns:
        Number of tokens produced by the configured tokenizer.
    """

    return len(encoding.encode(_coerce_optional_text(text, "text")))


def count_tokens_batch(texts: Sequence[str | None]) -> list[int]:
    """Count tokens for a batch of text strings.

    Batch tokenization is faster than repeatedly calling the tokenizer on one
    string at a time, especially when processing many reasoning blocks and
    summaries.

    Args:
        texts: Input strings. ``None`` values are treated as empty strings.

    Returns:
        List of token counts aligned with the input list.
    """

    validated_texts = [
        _coerce_optional_text(text, "texts") for text in texts
    ]
    return [len(tokens) for tokens in encoding.encode_batch(validated_texts)]


def parse_response(response: str | None) -> ParsedResponse:
    """Parse an OpenMementos response into reasoning and answer components.

    The parser assumes that the response contains a ``<think>...</think>``
    section. Inside that section, reasoning is represented as repeated
    ``block`` and ``summary`` spans.
    Missing responses are treated as empty strings.

    Args:
        response: Raw response string from OpenMementos.

    Returns:
        Dictionary containing the parsed thinking text, final answer text,
        extracted blocks, extracted summaries, block counts, summary counts,
        and character-length diagnostics.
    """

    response = _coerce_optional_text(response, "response")

    # reasoning chain is inside one <think>...</think> section
    think_match = THINK_RE.search(response)

    if think_match:
        think_text = think_match.group(1)
        answer_text = response[think_match.end():].strip()
    else:
        think_text = ""
        answer_text = response.strip()

    # extract aligned reasoning blocks and summaries in the thinking section
    blocks = [x.strip() for x in BLOCK_RE.findall(think_text)]
    summaries = [x.strip() for x in SUMMARY_RE.findall(think_text)]

    return {
        "think_text": think_text,
        "answer_text": answer_text,
        "blocks": blocks,
        "summaries": summaries,
        "n_blocks": len(blocks),
        "n_summaries": len(summaries),
        "think_chars": len(think_text),
        "answer_chars": len(answer_text),
        "response_chars": len(response),
    }


def extract_problem_features(problem: str | None) -> ProblemFeatures:
    """Extract lightweight predictor features from the original problem text.

    These features are safe predictors because they are computed only from the
    prompt/problem, before observing reasoning blocks, summaries, or compression
    outcomes, avoiding target leakage

    Args:
        problem: Raw problem text.

    Returns:
        Dictionary of problem-level features, including token length, character
        length, math-symbol share, multiple-choice indicator, code-fence
        indicator, and question-mark count.
    """

    problem = _coerce_optional_text(problem, "problem")

    n_chars = len(problem)
    n_math_symbols = sum(ch in MATH_SYMBOLS for ch in problem)

    return {
        "problem_chars": n_chars,
        "problem_tokens": count_tokens(problem),
        "problem_math_symbol_share": n_math_symbols / n_chars if n_chars else 0.0,
        "problem_has_multiple_choice": int(
            bool(re.search(r"\b[A-D][\).]", problem))
        ),
        "problem_has_code_fence": int("```" in problem),
        "problem_question_mark_count": problem.count("?"),
    }


def build_feature_tables(
    n_rows: int,
    dataset_id: str = DATASET_ID,
    split: str = SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build trace-level and block-level feature tables from a streamed sample.

    This helper is intended for exploratory and development notebooks. It
    streams only the first ``n_rows`` examples and materializes the resulting
    feature tables in memory.

    Args:
        n_rows: Number of streamed dataset rows to process.
        dataset_id: Hugging Face dataset identifier.
        split: Dataset split to stream.

    Returns:
        Tuple ``(df_traces, df_blocks)`` where ``df_traces`` has one row per
        reasoning trace and ``df_blocks`` has one row per block-summary pair.
    """
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, received {n_rows}.")

    ds_stream = load_dataset(dataset_id, split=split, streaming=True)
    return build_feature_chunk(list(islice(ds_stream, n_rows)), start_trace_id=0)


def extract_spacy_problem_features(doc: object) -> SpacyProblemFeatures:
    """Extract optional spaCy linguistic features from a parsed problem document.

    The input is a spaCy ``Doc`` object, not a raw string. These features are
    intended as optional problem-level predictors and are not required for the
    core token-compression pipeline.

    Args:
        doc: spaCy document produced from a problem string.

    Returns:
        Dictionary of linguistic features such as sentence count, average
        sentence length, stopword share, punctuation share, numeric-token share,
        out-of-vocabulary share, noun share, verb share, and entity count.
    """
    if not hasattr(doc, "sents") or not hasattr(doc, "ents"):
        raise TypeError("doc must be a spaCy Doc-like object with sents and ents.")

    tokens = [tok for tok in doc if not tok.is_space]

    if not tokens:
        return {
            "spacy_sentence_count": 0,
            "spacy_avg_sentence_tokens": 0,
            "spacy_stopword_share": 0,
            "spacy_punctuation_share": 0,
            "spacy_numeric_token_share": 0,
            "spacy_oov_share": 0,
            "spacy_noun_share": 0,
            "spacy_verb_share": 0,
            "spacy_entity_count": 0,
        }

    sentence_lengths = [
        len([tok for tok in sent if not tok.is_space])
        for sent in doc.sents
    ]

    return {
        "spacy_sentence_count": len(sentence_lengths),
        "spacy_avg_sentence_tokens": (
            sum(sentence_lengths) / len(sentence_lengths)
            if sentence_lengths
            else 0.0
        ),
        "spacy_stopword_share": sum(tok.is_stop for tok in tokens) / len(tokens),
        "spacy_punctuation_share": sum(tok.is_punct for tok in tokens) / len(tokens),
        "spacy_numeric_token_share": sum(tok.like_num for tok in tokens) / len(tokens),
        "spacy_oov_share": sum(tok.is_oov for tok in tokens) / len(tokens),
        "spacy_noun_share": sum(
            tok.pos_ in {"NOUN", "PROPN"} for tok in tokens
        ) / len(tokens),
        "spacy_verb_share": sum(
            tok.pos_ in {"VERB", "AUX"} for tok in tokens
        ) / len(tokens),
        "spacy_entity_count": len(doc.ents),
    }


def build_feature_chunk(
    rows: Sequence[Mapping[str, object]],
    start_trace_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature tables for one chunk of streamed dataset rows.

    This function is used by ``write_feature_partitions`` to process the full
    dataset in bounded-memory chunks.

    Args:
        rows: List of raw dataset rows.
        start_trace_id: Trace ID assigned to the first row in the chunk.

    Returns:
        Tuple ``(df_traces, df_blocks)`` for the chunk.
    """
    if start_trace_id < 0:
        raise ValueError(
            f"start_trace_id must be non-negative, received {start_trace_id}."
        )

    trace_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []

    for offset, row in enumerate(rows):
        trace_id = start_trace_id + offset
        _validate_dataset_row(row, row_index=trace_id)

        problem = _coerce_optional_text(row.get("problem"), "problem", trace_id)
        response = _coerce_optional_text(row.get("response"), "response", trace_id)
        metadata = _metadata_from_row(row, row_index=trace_id)
        parsed = parse_response(response)
        problem_features = extract_problem_features(problem)

        trace_rows.append({
            "trace_id": trace_id,
            **metadata,
            **problem_features,
            "response_chars": parsed["response_chars"],
            "response_tokens": count_tokens(response),
            "think_chars": parsed["think_chars"],
            "think_tokens": count_tokens(parsed["think_text"]),
            "answer_chars": parsed["answer_chars"],
            "answer_tokens": count_tokens(parsed["answer_text"]),
            "n_blocks": parsed["n_blocks"],
            "n_summaries": parsed["n_summaries"],
            "block_summary_delta": parsed["n_blocks"] - parsed["n_summaries"],
        })

        block_token_counts = count_tokens_batch(parsed["blocks"])
        summary_token_counts = count_tokens_batch(parsed["summaries"])

        for block_index, (block, summary, block_tokens, summary_tokens) in enumerate(
            zip(
                parsed["blocks"],
                parsed["summaries"],
                block_token_counts,
                summary_token_counts,
            )
        ):
            block_chars = len(block)
            summary_chars = len(summary)

            block_rows.append({
                "trace_id": trace_id,
                "block_index": block_index,
                **metadata,
                **problem_features,
                "n_blocks_in_trace": parsed["n_blocks"],
                "relative_block_position": (
                    block_index / (parsed["n_blocks"] - 1)
                    if parsed["n_blocks"] > 1
                    else 0
                ),
                "block_chars": block_chars,
                "summary_chars": summary_chars,
                "block_tokens": block_tokens,
                "summary_tokens": summary_tokens,
                "summary_to_block_char_ratio": _safe_ratio(
                    summary_chars,
                    block_chars,
                ),
                "summary_to_block_token_ratio": _safe_ratio(
                    summary_tokens,
                    block_tokens,
                ),
                "char_compression_savings": _safe_savings(
                    summary_chars,
                    block_chars,
                ),
                "token_compression_savings": _safe_savings(
                    summary_tokens,
                    block_tokens,
                ),
            })

    return pd.DataFrame(trace_rows), pd.DataFrame(block_rows)


def write_feature_partitions(
    output_dir: Path,
    chunk_size: int = 10_000,
    max_rows: int | None = None,
) -> BuildSummary:
    """Stream OpenMementos and write trace/block feature parquet partitions.

    The function writes two partitioned parquet directories under ``output_dir``:

    - ``traces/``: trace-level feature partitions
    - ``blocks/``: block-level feature partitions

    Args:
        output_dir: Directory where parquet partitions should be written.
        chunk_size: Number of original traces to process per parquet partition.
        max_rows: Optional maximum number of streamed rows. Use ``None`` to
            process the full split.

    Returns:
        Build summary with output path, number of trace rows, number of block
        rows, and number of written partitions.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, received {chunk_size}.")
    if max_rows is not None and max_rows < 0:
        raise ValueError(f"max_rows must be non-negative or None, received {max_rows}.")

    output_dir = Path(output_dir)
    trace_dir = output_dir / "traces"
    block_dir = output_dir / "blocks"
    trace_dir.mkdir(parents=True, exist_ok=True)
    block_dir.mkdir(parents=True, exist_ok=True)

    ds_stream = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    iterator = islice(ds_stream, max_rows) if max_rows is not None else ds_stream

    current_chunk: list[Mapping[str, object]] = []
    trace_id = 0
    part_id = 0
    total_blocks = 0

    for row in tqdm(iterator, desc="Streaming traces"):
        current_chunk.append(row)

        if len(current_chunk) >= chunk_size:
            df_traces, df_blocks = build_feature_chunk(current_chunk, trace_id)
            df_traces.to_parquet(
                trace_dir / f"traces_part_{part_id:04d}.parquet",
                index=False,
            )
            df_blocks.to_parquet(
                block_dir / f"blocks_part_{part_id:04d}.parquet",
                index=False,
            )

            trace_id += len(df_traces)
            total_blocks += len(df_blocks)
            part_id += 1
            current_chunk = []

    if current_chunk:
        df_traces, df_blocks = build_feature_chunk(current_chunk, trace_id)
        df_traces.to_parquet(
            trace_dir / f"traces_part_{part_id:04d}.parquet",
            index=False,
        )
        df_blocks.to_parquet(
            block_dir / f"blocks_part_{part_id:04d}.parquet",
            index=False,
        )

        trace_id += len(df_traces)
        total_blocks += len(df_blocks)
        part_id += 1

    return {
        "output_dir": str(output_dir),
        "n_trace_rows": trace_id,
        "n_block_rows": total_blocks,
        "n_parts": part_id,
    }


def latest_feature_build_dir(base_dir: Path) -> Path:
    """Return the newest local full-feature build directory.

    The project keeps generated parquet files under ignored ``data/`` paths.
    This helper lets notebooks use an existing local build without hard-coding a
    timestamped run directory or triggering a fresh dataset stream.
    """

    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Feature build directory does not exist: {base_dir}")

    candidates = [
        path for path in base_dir.iterdir()
        if path.is_dir()
        and all((path / file_name).exists() for file_name in FEATURE_BUILD_FILES)
    ]
    if not candidates:
        required = ", ".join(FEATURE_BUILD_FILES)
        raise FileNotFoundError(
            f"No complete feature build found under {base_dir}. "
            f"Expected each build directory to contain: {required}."
        )

    return max(candidates, key=lambda path: path.name)


def count_share_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count rows and percentage shares for a dataframe column.

    Args:
        df: Input dataframe.
        column: Column name to summarize.

    Returns:
        DataFrame with row counts and percentage shares for each value.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe.")

    counts = df[column].value_counts(dropna=False)
    shares = df[column].value_counts(normalize=True, dropna=False)

    return (
        pd.DataFrame({"n_rows": counts, "share": shares})
        .assign(share_pct=lambda x: (x["share"] * 100).round(2))
        .drop(columns="share")
    )


def format_describe(summary: pd.Series, decimals: int = 4) -> pd.Series:
    """Format a pandas ``describe`` summary for readable notebook display.

    The ``count`` row is formatted as an integer with thousands separators.
    Other rows are formatted as fixed-decimal values.

    Args:
        summary: Series returned by ``pandas.Series.describe``.
        decimals: Number of decimal places for non-count rows.

    Returns:
        Formatted summary series with string values.
    """
    if "count" not in summary.index:
        raise KeyError("summary must contain a 'count' index entry.")

    # Keep count readable as an integer and format distribution statistics.
    formatted = summary.copy().astype(object)
    formatted["count"] = f"{int(summary['count']):,}"

    for idx in summary.index.drop("count"):
        formatted[idx] = f"{summary[idx]:.{decimals}f}"

    return formatted
