## Handles tokenization, sequence creation, and preprocessing of raw text into fixed-length token sequences. 

from transformers import AutoTokenizer
from itertools import chain


def get_tokenizer():
    """
    Load BERT tokenizer.
    """
    print("[INFO] Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("[INFO] Tokenizer loaded.")
    return tokenizer


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize dataset text into token IDs.
    """

    def tokenize_function(example):
        return tokenizer(
            example["text"],  ## input raw strings "This is a sentence."
            add_special_tokens=True, ## add BERT special tokens [CLS], [SEP]. [CLS] This is a sentence . [SEP]
            truncation=False, ## keep full tokenized length
        )

    print("[INFO] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True, ## processes multiple examples per batch for speed
        remove_columns=["text"], ## removes original text columns and keep input_ids and attention_mask only
    )

    return tokenized_dataset


def create_fixed_length_sequences(tokenized_dataset, seq_len=256, stride=64):
    """
    Create overlapping sequences using sliding window to reach the project requirements

    seq_len: length of each sequence
    stride: step size between windows (smaller stride = more data)
    """

    print(f"[INFO] Creating fixed-length sequences (seq_len={seq_len}, stride={stride})...")

    all_tokens = []

    # Concatenate all input_ids
    for example in tokenized_dataset:
        all_tokens.extend(example["input_ids"])

    sequences = []

    # Sliding window
    for start_idx in range(0, len(all_tokens) - seq_len, stride):
        end_idx = start_idx + seq_len
        sequences.append(all_tokens[start_idx:end_idx])

    print(f"[INFO] Total sequences created: {len(sequences)}")

    return sequences