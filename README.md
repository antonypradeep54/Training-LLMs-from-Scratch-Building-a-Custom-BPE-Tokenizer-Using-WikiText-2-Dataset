# Custom BPE Tokenizer on WikiText-2

This project trains a custom Byte Pair Encoding (BPE) tokenizer from scratch on the WikiText-2 subset of the `salesforce/wikitext` dataset (config `wikitext-2-v1`) and demonstrates saving, reloading, and evaluating it.

Contents
- `train_tokenizer.py` — loads & cleans WikiText-2, trains a BPE tokenizer, evaluates it on val/test, saves tokenizer in HuggingFace-compatible format.
- `requirements.txt` — Python dependencies.
- `README.md` — this file.

Quick summary of features
- Deduplicates training text and normalizes lines.
- Trains a BPE tokenizer (default vocab_size=30000) with special tokens: `[PAD] [UNK] [CLS] [SEP] [MASK]`.
- Evaluates: vocabulary size, average tokens per sentence, unk rate, tokenization consistency (encode->decode match fraction), and compression ratio (avg chars per token).
- Saves tokenizer to a directory that can be loaded as a `transformers.PreTrainedTokenizerFast`.

Requirements
- Python 3.9+ (3.10 recommended)
- See `requirements.txt` for pip packages.

Example usage
1. Create a virtual env:
   python -m venv .venv
   source .venv/bin/activate

2. Install requirements:
   pip install -r requirements.txt

3. Train the tokenizer (defaults shown):
   python train_tokenizer.py \
     --vocab_size 30000 \
     --output_dir ./tokenizer-wikitext2 \
     --special_tokens "[PAD],[UNK],[CLS],[SEP],[MASK]" \
     --max_train_lines 200000

Notes:
- `max_train_lines` controls the number of deduplicated training lines to use (helps for faster iteration). Set to `0` to use all.
- Training is CPU-bound. GPU is not required.

RunPod VM recommendations
- Tokenizer training is CPU & I/O bound; GPU is not required.
- Recommended minimal configuration for interactive work:
  - CPU: 8 vCPUs (higher is better for faster training)
  - RAM: 32 GB
  - Disk: 50+ GB SSD
  - OS: Ubuntu 22.04
  - Python: 3.10
- If you plan to fine-tune or run a language model afterward, add a GPU instance:
  - For small models: 1x NVIDIA A10 / T4 (16–24 GB)
  - For larger models: 1x NVIDIA A100 (40/80 GB)
- Networking: Make sure outbound access to Hugging Face is allowed (dataset download).

What I provide
- A single Python script `train_tokenizer.py` that:
  - Loads the dataset
  - Cleans & deduplicates
  - Trains a BPE tokenizer
  - Evaluates on validation and test
  - Saves a HF-compatible tokenizer directory with `tokenizer.json` plus files created by `transformers.PreTrainedTokenizerFast.save_pretrained`.

If you'd like:
- I can split training & evaluation into separate files.
- I can add a small Jupyter notebook demonstrating interactive usage and visualization of token-length distributions.