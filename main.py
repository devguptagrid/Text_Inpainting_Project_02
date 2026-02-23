from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.dataset import TextInpaintingDataset
from models.transformer import TransformerDenoiser

from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    dataset = load_wikitext()
    train_dataset = clean_dataset(dataset["train"])

    tokenizer = get_tokenizer()

    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    train_sequences = create_fixed_length_sequences(
        tokenized_train,
        seq_len=256
    )

    train_data = TextInpaintingDataset(
        sequences=train_sequences,
        tokenizer=tokenizer,
        mask_type="span",
        mask_ratio=0.25,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
    )

    # Initialize model
    model = TransformerDenoiser().to(device)

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"].to(device)

    logits = model(input_ids)

    print("Input shape:", input_ids.shape)
    print("Logits shape:", logits.shape)