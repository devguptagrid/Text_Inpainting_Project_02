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
            example["text"],
            add_special_tokens=True,
            truncation=False,
        )

    print("[INFO] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    return tokenized_dataset


def create_fixed_length_sequences(tokenized_dataset, seq_len=256):
    """
    Concatenate all tokenized inputs and split into fixed-length chunks.
    """

    print(f"[INFO] Creating fixed-length sequences (seq_len={seq_len})...")

    # Flatten list of token lists
    all_tokens = list(chain(*tokenized_dataset["input_ids"]))

    # Drop remainder tokens that don't fit into full seq_len
    total_length = len(all_tokens)
    total_length = (total_length // seq_len) * seq_len

    all_tokens = all_tokens[:total_length]

    # Create chunks
    sequences = [
        all_tokens[i : i + seq_len]
        for i in range(0, total_length, seq_len)
    ]

    print(f"[INFO] Total sequences created: {len(sequences)}")

    return sequences