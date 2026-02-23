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

    all_tokens = all_tokens[:total_length] ## removes leftover tokens that don't fit into a full sequence

    # Create chunks
    sequences = [
        all_tokens[i : i + seq_len]
        for i in range(0, total_length, seq_len)
    ]

    print(f"[INFO] Total sequences created: {len(sequences)}")

    return sequences