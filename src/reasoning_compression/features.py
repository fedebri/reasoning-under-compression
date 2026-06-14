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

from itertools import islice
from pathlib import Path
import re

import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm


DATASET_ID = "microsoft/OpenMementos"
SPLIT = "train"
ENCODING_NAME = "cl100k_base"

BLOCK_RE = re.compile(r"<\|block_start\|>(.*?)<\|block_end\|>", re.DOTALL)
SUMMARY_RE = re.compile(r"<\|summary_start\|>(.*?)<\|summary_end\|>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
MATH_SYMBOLS = set("+-=*/^<>≤≥≈≠√∑∫π%()[]{}")

encoding = tiktoken.get_encoding(ENCODING_NAME)


__all__ = [
    "DATASET_ID",
    "SPLIT",
    "ENCODING_NAME",
    "count_tokens",
    "count_tokens_batch",
    "parse_response",
    "extract_problem_features",
    "build_feature_tables",
    "extract_spacy_problem_features",
    "build_feature_chunk",
    "write_feature_partitions",
    "format_metrics_percent",
    "count_share_table",
    "format_describe",
]


def count_tokens(text: str) -> int:
    """Count tokens in a text string using the configured tiktoken encoding.

    Args:
        text: Input text. ``None`` is treated as an empty string.

    Returns:
        Number of tokens produced by the configured tokenizer.
    """

    if text is None:
        return 0
    return len(encoding.encode(text))


def count_tokens_batch(texts: list[str]) -> list[int]:
    """Count tokens for a batch of text strings.

    Batch tokenization is faster than repeatedly calling the tokenizer on one
    string at a time, especially when processing many reasoning blocks and
    summaries.

    Args:
        texts: List of input strings. ``None`` values are treated as empty strings.

    Returns:
        List of token counts aligned with the input list.
    """

    texts = ["" if x is None else x for x in texts]
    return [len(tokens) for tokens in encoding.encode_batch(texts)]


# helper function to preprocess the response variable
def parse_response(response: str) -> dict:
    """Parse an OpenMementos response into reasoning and answer components.

    The parser assumes that the response contains a ``<think>...</think>``
    section. Inside that section, reasoning is represented as repeated
    ``block`` and ``summary`` spans.
    Eventual missing responses are treated as empty strings

    Args:
        response: Raw response string from OpenMementos.

    Returns:
        Dictionary containing the parsed thinking text, final answer text,
        extracted blocks, extracted summaries, block counts, summary counts,
        and character-length diagnostics.
    """

    if response is None:
        response = ""

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




def extract_problem_features(problem: str) -> dict:
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

    if problem is None:
        problem = ""

    n_chars = len(problem)
    n_math_symbols = sum(ch in MATH_SYMBOLS for ch in problem)

    return {
        "problem_chars": n_chars,
        "problem_tokens": count_tokens(problem),
        "problem_math_symbol_share": n_math_symbols / n_chars if n_chars else 0,
        "problem_has_multiple_choice": int(bool(re.search(r"\b[A-D][\).]", problem))),
        "problem_has_code_fence": int("```" in problem),
        "problem_question_mark_count": problem.count("?"),
    }




def build_feature_tables(n_rows: int, dataset_id: str = DATASET_ID, split: str = SPLIT,) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    ds_stream = load_dataset(dataset_id, split=split, streaming=True)

    trace_rows = []
    block_rows = []

    for trace_id, row in enumerate(islice(ds_stream, n_rows)):
        problem = row.get("problem") or ""
        response = row.get("response") or ""
        parsed = parse_response(response)
        problem_features = extract_problem_features(problem)

        trace_rows.append({
            "trace_id": trace_id,
            "domain": row.get("domain"),
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
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
            zip(parsed["blocks"], parsed["summaries"], block_token_counts, summary_token_counts)
        ):
            block_chars = len(block)
            summary_chars = len(summary)

            block_rows.append({
                "trace_id": trace_id,
                "block_index": block_index,
                "domain": row.get("domain"),
                "source": row.get("source"),
                "difficulty": row.get("difficulty"),
                **problem_features,
                "n_blocks_in_trace": parsed["n_blocks"],
                "relative_block_position": block_index / (parsed["n_blocks"] - 1) if parsed["n_blocks"] > 1 else 0,
                "block_chars": block_chars,
                "summary_chars": summary_chars,
                "block_tokens": block_tokens,
                "summary_tokens": summary_tokens,
                "summary_to_block_char_ratio": summary_chars / block_chars if block_chars else np.nan,
                "summary_to_block_token_ratio": summary_tokens / block_tokens if block_tokens else np.nan,
                "char_compression_savings": 1 - (summary_chars / block_chars) if block_chars else np.nan,
                "token_compression_savings": 1 - (summary_tokens / block_tokens) if block_tokens else np.nan,
            })

    return pd.DataFrame(trace_rows), pd.DataFrame(block_rows)


def extract_spacy_problem_features(doc: object) -> dict:
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
    # spaCy features for the original problem text
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

    sentence_lengths = [len([tok for tok in sent if not tok.is_space]) for sent in doc.sents]

    return {
        "spacy_sentence_count": len(sentence_lengths),
        "spacy_avg_sentence_tokens": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
        "spacy_stopword_share": sum(tok.is_stop for tok in tokens) / len(tokens),
        "spacy_punctuation_share": sum(tok.is_punct for tok in tokens) / len(tokens),
        "spacy_numeric_token_share": sum(tok.like_num for tok in tokens) / len(tokens),
        "spacy_oov_share": sum(tok.is_oov for tok in tokens) / len(tokens),
        "spacy_noun_share": sum(tok.pos_ in {"NOUN", "PROPN"} for tok in tokens) / len(tokens),
        "spacy_verb_share": sum(tok.pos_ in {"VERB", "AUX"} for tok in tokens) / len(tokens),
        "spacy_entity_count": len(doc.ents),
    }




def build_feature_chunk(rows: list[dict], start_trace_id: int,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature tables for one chunk of streamed dataset rows.

    This function is used by ``write_feature_partitions`` to process the full
    dataset in bounded-memory chunks.

    Args:
        rows: List of raw dataset rows.
        start_trace_id: Trace ID assigned to the first row in the chunk.

    Returns:
        Tuple ``(df_traces, df_blocks)`` for the chunk.
    """
    trace_rows = []
    block_rows = []

    for offset, row in enumerate(rows):
        trace_id = start_trace_id + offset
        problem = row.get("problem") or ""
        response = row.get("response") or ""
        parsed = parse_response(response)
        problem_features = extract_problem_features(problem)

        trace_rows.append({
            "trace_id": trace_id,
            "domain": row.get("domain"),
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
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
            zip(parsed["blocks"], parsed["summaries"], block_token_counts, summary_token_counts)
        ):
            block_chars = len(block)
            summary_chars = len(summary)

            block_rows.append({
                "trace_id": trace_id,
                "block_index": block_index,
                "domain": row.get("domain"),
                "source": row.get("source"),
                "difficulty": row.get("difficulty"),
                **problem_features,
                "n_blocks_in_trace": parsed["n_blocks"],
                "relative_block_position": block_index / (parsed["n_blocks"] - 1) if parsed["n_blocks"] > 1 else 0,
                "block_chars": block_chars,
                "summary_chars": summary_chars,
                "block_tokens": block_tokens,
                "summary_tokens": summary_tokens,
                "summary_to_block_char_ratio": summary_chars / block_chars if block_chars else np.nan,
                "summary_to_block_token_ratio": summary_tokens / block_tokens if block_tokens else np.nan,
                "char_compression_savings": 1 - (summary_chars / block_chars) if block_chars else np.nan,
                "token_compression_savings": 1 - (summary_tokens / block_tokens) if block_tokens else np.nan,
            })

    return pd.DataFrame(trace_rows), pd.DataFrame(block_rows)




def write_feature_partitions(output_dir: Path, chunk_size: int = 10_000, max_rows: int | None = None,) -> dict:
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

    trace_dir = output_dir / "traces"
    block_dir = output_dir / "blocks"
    trace_dir.mkdir(parents=True, exist_ok=True)
    block_dir.mkdir(parents=True, exist_ok=True)

    ds_stream = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    iterator = islice(ds_stream, max_rows) if max_rows is not None else ds_stream

    current_chunk = []
    trace_id = 0
    part_id = 0
    total_blocks = 0

    for row in tqdm(iterator, desc="Streaming traces"):
        current_chunk.append(row)

        if len(current_chunk) >= chunk_size:
            df_traces, df_blocks = build_feature_chunk(current_chunk, trace_id)
            df_traces.to_parquet(trace_dir / f"traces_part_{part_id:04d}.parquet", index=False)
            df_blocks.to_parquet(block_dir / f"blocks_part_{part_id:04d}.parquet", index=False)

            trace_id += len(df_traces)
            total_blocks += len(df_blocks)
            part_id += 1
            current_chunk = []

    if current_chunk:
        df_traces, df_blocks = build_feature_chunk(current_chunk, trace_id)
        df_traces.to_parquet(trace_dir / f"traces_part_{part_id:04d}.parquet", index=False)
        df_blocks.to_parquet(block_dir / f"blocks_part_{part_id:04d}.parquet", index=False)

        trace_id += len(df_traces)
        total_blocks += len(df_blocks)
        part_id += 1

    return {
        "output_dir": str(output_dir),
        "n_trace_rows": trace_id,
        "n_block_rows": total_blocks,
        "n_parts": part_id,
    }


def format_metrics_percent(metrics: dict) -> pd.DataFrame:
    """Format model metrics as a compact percentage dataframe.

    Args:
        metrics: Dictionary mapping metric names to decimal values, such as
            ``0.802`` for 80.2%.

    Returns:
        DataFrame with ``metric`` and formatted percentage ``value`` columns.
    """
    return pd.DataFrame({
        "metric": metrics.keys(),
        "value": [f"{value * 100:.2f}%" for value in metrics.values()],
    })


def count_share_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count rows and percentage shares for a dataframe column.

    Args:
        df: Input dataframe.
        column: Column name to summarize.

    Returns:
        DataFrame with row counts and percentage shares for each value.
    """
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
    # Keep count readable as an integer while formatting distribution statistics
    # as fixed-decimal values.
    formatted = summary.copy().astype(object)
    formatted["count"] = f"{int(summary['count']):,}"

    for idx in summary.index.drop("count"):
        formatted[idx] = f"{summary[idx]:.{decimals}f}"

    return formatted