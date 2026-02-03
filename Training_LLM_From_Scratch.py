#!/usr/bin/env python3
"""
Train a BPE tokenizer on WikiText-2 (salesforce/wikitext, config: wikitext-2-v1),
evaluate on validation and test, and save a HuggingFace-compatible tokenizer.

Usage:
    python train_tokenizer.py \
        --vocab_size 30000 \
        --output_dir ./tokenizer-wikitext2 \
        --special_tokens "[PAD],[UNK],[CLS],[SEP],[MASK]" \
        --max_train_lines 200000
"""

import argparse
import os
import json
from typing import Iterable, List, Tuple
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np


def load_and_clean_dataset(config_name="wikitext-2-v1"):
    ds = load_dataset("salesforce/wikitext", config_name)
    # splits: 'train', 'validation', 'test'
    return ds


def deduplicate_and_normalize(lines: Iterable[str]) -> List[str]:
    seen = set()
    cleaned = []
    for ln in lines:
        if ln is None:
            continue
        s = ln.replace("<unk>", " ")  # remove exact token occurrences
        s = " ".join(s.split())  # collapse whitespace
        s = s.strip()
        if s == "":
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


def train_bpe_tokenizer(text_iterator: Iterable[str],
                        vocab_size: int = 30000,
                        special_tokens: List[str] = None,
                        show_progress: bool = True) -> Tokenizer:
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    # Create empty BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # Normalizer and pre-tokenizer
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    # train_from_iterator accepts any iterable of strings
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return tokenizer


def evaluate_tokenizer_fast(pretrained_tokenizer: PreTrainedTokenizerFast,
                            texts: Iterable[str],
                            sample_limit: int = 0) -> dict:
    """
    Evaluate tokenizer on provided texts.
    sample_limit: if >0, limit the number of lines to evaluate (random sample).
    Returns metrics dict.
    """
    # collect lines
    lines = [ln for ln in texts if (ln is not None and ln.strip() != "")]
    if sample_limit and sample_limit > 0 and len(lines) > sample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(lines), size=sample_limit, replace=False)
        lines = [lines[i] for i in idx]

    total_tokens = 0
    total_chars = 0
    total_sentences = 0
    total_unk_tokens = 0
    encode_decode_matches = 0

    unk_id = pretrained_tokenizer.unk_token_id

    for ln in lines:
        total_sentences += 1
        chars = len(ln)
        total_chars += chars
        # encode without adding special tokens for per-sentence metric
        #ids = pretrained_tokenizer.encode(ln, add_special_tokens=False).ids
        ids = pretrained_tokenizer.encode(ln, add_special_tokens=False)
        total_tokens += len(ids)
        if unk_id is not None:
            total_unk_tokens += sum(1 for i in ids if i == unk_id)
        # check encode->decode match (normalized compare)
        decoded = pretrained_tokenizer.decode(ids, skip_special_tokens=True).strip()
        # Normalize both sides for loose equality
        if " ".join(ln.split()) == " ".join(decoded.split()):
            encode_decode_matches += 1

    avg_tokens_per_sentence = (total_tokens / total_sentences) if total_sentences else 0.0
    unk_rate = (total_unk_tokens / total_tokens) if total_tokens else 0.0
    compression = (total_chars / total_tokens) if total_tokens else 0.0
    consistency = (encode_decode_matches / total_sentences) if total_sentences else 0.0

    return {
        "num_sentences": total_sentences,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "avg_tokens_per_sentence": float(avg_tokens_per_sentence),
        "unk_rate": float(unk_rate),
        "chars_per_token": float(compression),
        "encode_decode_consistency": float(consistency),
    }


def save_hf_tokenizer(tokenizer: Tokenizer, output_dir: str, special_tokens: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    tok_json = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tok_json)

    # Build PreTrainedTokenizerFast wrapper with tokenizers' tokenizer.json
    # Map special tokens to names
    pad_token, unk_token, cls_token, sep_token, mask_token = special_tokens
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_json,
                                           unk_token=unk_token,
                                           pad_token=pad_token,
                                           cls_token=cls_token,
                                           sep_token=sep_token,
                                           mask_token=mask_token)
    # Save in HF format (creates config + merges/vocab if necessary)
    hf_tokenizer.save_pretrained(output_dir)
    return hf_tokenizer


def parse_special_tokens_arg(s: str) -> List[str]:
    # Accept comma-separated special tokens string
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 5:
        raise ValueError("Please provide 5 special tokens in the order: [PAD],[UNK],[CLS],[SEP],[MASK]")
    return parts[:5]


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on WikiText-2")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--output_dir", type=str, default="./tokenizer-wikitext2")
    parser.add_argument("--special_tokens", type=str, default="[PAD],[UNK],[CLS],[SEP],[MASK]",
                        help="Comma separated list of 5 special tokens in order PAD,UNK,CLS,SEP,MASK")
    parser.add_argument("--max_train_lines", type=int, default=0,
                        help="Max number of deduplicated training lines to use. 0 = all.")
    parser.add_argument("--sample_eval_lines", type=int, default=20000,
                        help="Max number of lines to use for eval per split (0 = all).")
    args = parser.parse_args()

    special_tokens = parse_special_tokens_arg(args.special_tokens)

    print("Loading dataset...")
    ds = load_and_clean_dataset()
    print("Cleaning and deduplicating training data (this may take a while)...")
    train_lines = deduplicate_and_normalize(ds["train"]["text"])
    if args.max_train_lines and args.max_train_lines > 0:
        train_lines = train_lines[: args.max_train_lines]
    print(f"Training lines (deduplicated): {len(train_lines)}")

    print("Training BPE tokenizer...")
    tokenizer = train_bpe_tokenizer(train_lines,
                                    vocab_size=args.vocab_size,
                                    special_tokens=special_tokens,
                                    show_progress=True)

    print("Saving tokenizer in HuggingFace format...")
    hf_tokenizer = save_hf_tokenizer(tokenizer, args.output_dir, special_tokens)
    print(f"Saved tokenizer to: {args.output_dir}")

    # Report vocabulary size
    try:
        vocab_size = hf_tokenizer.vocab_size
    except Exception:
        # Fallback: read from tokenizer.json
        tok_json_path = os.path.join(args.output_dir, "tokenizer.json")
        with open(tok_json_path, "r", encoding="utf-8") as f:
            tokj = json.load(f)
        # tokenizers vocab size location is model.vocab or merges; use lengths
        vocab_size = len(tokj.get("model", {}).get("vocab", {})) or args.vocab_size

    print(f"Report: Requested vocab_size={args.vocab_size}, actual vocab_size={vocab_size}")

    # Evaluate on validation and test splits
    print("Evaluating on validation split...")
    val_metrics = evaluate_tokenizer_fast(hf_tokenizer,
                                         ds["validation"]["text"],
                                         sample_limit=args.sample_eval_lines)
    print("Evaluating on test split...")
    test_metrics = evaluate_tokenizer_fast(hf_tokenizer,
                                          ds["test"]["text"],
                                          sample_limit=args.sample_eval_lines)

    # Print metrics summary
    def print_metrics(prefix: str, metrics: dict):
        print(f"--- {prefix} ---")
        print(f"Num sentences evaluated: {metrics['num_sentences']}")
        print(f"Total tokens: {metrics['total_tokens']}")
        print(f"Total chars: {metrics['total_chars']}")
        print(f"Average tokens per sentence: {metrics['avg_tokens_per_sentence']:.4f}")
        print(f"UNK rate: {metrics['unk_rate']:.6f}")
        print(f"Chars per token (compression): {metrics['chars_per_token']:.4f}")
        print(f"Encode->Decode consistency: {metrics['encode_decode_consistency']:.6f}")
        print("")

    print_metrics("VALIDATION", val_metrics)
    print_metrics("TEST", test_metrics)

    # Demo encode/decode
    demo_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing with BPE tokenization!",
        "This is an example line from WikiText-2 dataset."
    ]
    print("Demo encodings:")
    for t in demo_texts:
        enc = hf_tokenizer.encode(t)
        print(f"Text: {t}")
        print(f"Tokens: {enc.tokens}")
        print(f"IDs: {enc.ids}")
        dec = hf_tokenizer.decode(enc.ids)
        print(f"Decoded: {dec}")
        print("")


if __name__ == "__main__":
    main()
