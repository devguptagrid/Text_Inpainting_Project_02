from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.dataset import TextInpaintingDataset
from models.transformer import TransformerDenoiser
from training.trainer import train_one_epoch

from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    # Load dataset
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

    model = TransformerDenoiser().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    print("Starting training...")

    avg_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
    )

    print("Average Loss after 1 epoch:", avg_loss)