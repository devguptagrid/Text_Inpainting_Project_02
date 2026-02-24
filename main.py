from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.dataset import TextInpaintingDataset
from models.transformer import TransformerDenoiser
from training.trainer import train_one_epoch, evaluate

from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    dataset = load_wikitext()

    train_dataset = clean_dataset(dataset["train"])
    val_dataset = clean_dataset(dataset["validation"])

    tokenizer = get_tokenizer()

    ##Tokenize train
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    train_sequences = create_fixed_length_sequences(
        tokenized_train,
        seq_len=256
    )

    # Tokenize validation
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    val_sequences = create_fixed_length_sequences(
        tokenized_val,
        seq_len=256
    )

    train_data = TextInpaintingDataset(
    sequences=train_sequences,
    tokenizer=tokenizer,
    mask_type="span",
    mask_ratio=0.25,
    dynamic_masking=True,   # training = dynamic
)

    val_data = TextInpaintingDataset(
        sequences=val_sequences,
        tokenizer=tokenizer,
        mask_type="span",
        mask_ratio=0.25,
        dynamic_masking=False,  # validation = fixed
    )


    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
    )

    model = TransformerDenoiser(
    hidden_dim=384,
    num_layers=6,
    num_heads=6,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    num_epochs = 5

    print("\nStarting Training...\n")

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

    