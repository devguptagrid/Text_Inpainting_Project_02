from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.dataset import TextInpaintingDataset
from models.transformer import BertDenoiser
from training.trainer import train_one_epoch, evaluate

from torch.utils.data import DataLoader
import torch

from models.diffusion_model import DiffusionBert
from diffusion.forward_process import DiscreteDiffusionForward
from training.diffusion_trainer import train_diffusion_epoch, evaluate_diffusion
from data.diffusion_dataset import DiffusionDataset

mode = "baseline"   # "baseline" or "diffusion"

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
        seq_len=256,
        stride=32   # to get 50k+sequences
)

    # Tokenize validation
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    val_sequences = create_fixed_length_sequences(
        tokenized_val,
        seq_len=256,
        stride=32
    )
    num_epochs = 3

    if mode == "baseline":
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

        

    

        print("\nRunning BASELINE training...\n")

        model = BertDenoiser().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )

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


    elif mode == "diffusion":

        print("\nRunning DIFFUSION training...\n")
        
        T = 8
        train_data = DiffusionDataset(train_sequences)
        val_data = DiffusionDataset(val_sequences)

        train_loader = DataLoader(
            train_data,
            batch_size=16,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=16,
            shuffle=False,
        )

        model = DiffusionBert(T=T).to(device)

        diffusion_forward = DiscreteDiffusionForward(
            T=T,
            mask_token_id=tokenizer.mask_token_id
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-5,
        )

        for epoch in range(num_epochs):

            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_diffusion_epoch(
                model,
                train_loader,
                optimizer,
                diffusion_forward,
                tokenizer,
                device,
            )

            val_loss, val_acc = evaluate_diffusion(
                model,
                val_loader,
                diffusion_forward,
                tokenizer,
                device,
            )

            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")