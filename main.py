from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.dataset import TextInpaintingDataset


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

    # Create dataset with span masking
    train_data = TextInpaintingDataset(
        sequences=train_sequences,
        tokenizer=tokenizer,
        mask_type="span",
        mask_ratio=0.25,
    )

    print("Dataset size:", len(train_data))

    sample = train_data[0]

    print("Input shape:", sample["input_ids"].shape)
    print("Target shape:", sample["target_ids"].shape)
    print("Mask positions sum:", sample["mask_positions"].sum().item())