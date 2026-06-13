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


def count_tokens(text: str) -> int:
    if text is None:
        return 0
    return len(encoding.encode(text))


def count_tokens_batch(texts: list[str]) -> list[int]:
    # Batch tokenization is faster than encoding block-summary pairs one at a time.
    texts = ["" if x is None else x for x in texts]
    return [len(tokens) for tokens in encoding.encode_batch(texts)]


# helper function to preprocess the response variable
def parse_response(response: str) -> dict:
    # eventual missing responses are treated as empty strings
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
    # predictors come only from the original problem, avoiding target leakage
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


def build_feature_tables(n_rows: int, dataset_id: str = DATASET_ID, split: str = SPLIT):
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


def extract_spacy_problem_features(doc) -> dict:
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


def build_feature_chunk(rows: list[dict], start_trace_id: int):
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


def write_feature_partitions(output_dir: Path, chunk_size: int = 10_000, max_rows: int | None = None):
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
    return pd.DataFrame({
        "metric": metrics.keys(),
        "value": [f"{value * 100:.2f}%" for value in metrics.values()],
    })


def count_share_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False)
    shares = df[column].value_counts(normalize=True, dropna=False)

    return (
        pd.DataFrame({"n_rows": counts, "share": shares})
        .assign(share_pct=lambda x: (x["share"] * 100).round(2))
        .drop(columns="share")
    )
