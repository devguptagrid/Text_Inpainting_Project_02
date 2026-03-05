##Loads the WikiText dataset and prepares it for preprocessing and tokenization.

from datasets import load_dataset


def load_wikitext():
    """
    Loads WikiText-2 raw dataset from HuggingFace.
    Returns train, validation, test splits.
    """

    print("[INFO] Loading WikiText-2 dataset...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("[INFO] Dataset loaded successfully!")
    print(dataset)

    return dataset


def clean_dataset(dataset): 
    """
    Removes empty lines and very short sequences of less than 10 characters
    """

    def remove_empty(example):
        return example["text"] is not None and len(example["text"].strip()) > 10

    print("[INFO] Cleaning dataset (removing empty lines)...")

    cleaned = dataset.filter(remove_empty)

    print("[INFO] Cleaning complete.")
    return cleaned