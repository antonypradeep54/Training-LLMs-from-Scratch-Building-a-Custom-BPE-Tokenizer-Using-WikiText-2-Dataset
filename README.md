# Custom BPE Tokenizer on WikiText-2

This project trains a custom Byte Pair Encoding (BPE) tokenizer on the WikiText-2 subset of the `salesforce/wikitext` dataset (configuration `wikitext-2-v1`). It evaluates the tokenizer on validation and test text and saves it in a Hugging Face-compatible format.

**Scope:** this project trains a tokenizer; it does not train a language model.

## Contents

- `train_tokenizer.py` — dataset cleaning, BPE tokenizer training, evaluation, and saving
- `requirements.txt` — Python dependencies
- `README.md` — project documentation

## Features

- Deduplicates and normalizes training text.
- Trains a BPE tokenizer with a default vocabulary size of 30,000.
- Supports special tokens `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, and `[MASK]`.
- Reports vocabulary size, average tokens per sentence, unknown-token rate, encode/decode consistency, and characters per token on evaluation text.
- Saves a tokenizer that can be loaded with Hugging Face Transformers.

## Requirements

- Python 3.9+ (Python 3.10 recommended)
- See `requirements.txt` for Python packages.
- Outbound network access is needed to download the dataset from Hugging Face.

## Run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Train the tokenizer:

```bash
python train_tokenizer.py \
  --vocab_size 30000 \
  --output_dir ./tokenizer-wikitext2 \
  --special_tokens "[PAD],[UNK],[CLS],[SEP],[MASK]" \
  --max_train_lines 200000
```

The `--max_train_lines` argument limits the number of deduplicated training lines used. Set it to `0` to use all available deduplicated training lines. Evaluation uses up to 20,000 lines per validation/test split by default; this can be adjusted with `--sample_eval_lines`.

## Output

The output directory contains `tokenizer.json` and the files written by `PreTrainedTokenizerFast.save_pretrained`. The script also prints evaluation summaries for the validation and test splits.

## Limitations

Reported metrics describe this tokenizer on the selected WikiText-2 evaluation text. They are not evidence of language-model training quality or downstream task performance. Actual results depend on the dataset, vocabulary size, and evaluation settings.
