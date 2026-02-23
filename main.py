from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.dataset import TextInpaintingDataset

from torch.utils.data import DataLoader


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    # Load dataset
    dataset = load_wikitext()
    train_dataset = clean_dataset(dataset["train"])

    # Tokenizer
    tokenizer = get_tokenizer()

    # Tokenize and create sequences
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    train_sequences = create_fixed_length_sequences(
        tokenized_train,
        seq_len=256
    )

    # Create PyTorch dataset
    train_data = TextInpaintingDataset(
        sequences=train_sequences,
        tokenizer=tokenizer,
        mask_type="span",
        mask_ratio=0.25,
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
    )

    print("Number of batches per epoch:", len(train_loader))

    # Get one batch
    batch = next(iter(train_loader))

    print("Batch input shape:", batch["input_ids"].shape)
    print("Batch target shape:", batch["target_ids"].shape)
    print("Batch mask shape:", batch["mask_positions"].shape)