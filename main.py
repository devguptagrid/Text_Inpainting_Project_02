## Main script to run training, evaluation, and inference for both baseline and diffusion models. It loads the dataset, preprocesses it, 
# initializes the model and optimizer, and runs the training loop. It also includes code for evaluating the model on the test set and 
# computing BLEU/ROUGE scores.

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

from evaluation.bleu import compute_masked_bleu
from evaluation.rouge import compute_masked_rouge_l
from inference.reverse_diffusion import reverse_diffusion_sample

mode = "test"   # "baseline" or "diffusion" or "inference" or "test"

if __name__ == "__main__":
    set_seed(42) ## Set random seed for reproducibility across runs, ensuring that the same sequence of random numbers is generated each time the code is executed, 
    device = get_device() ## Detects if a CPU or Apple Silicon (MPS) is available and returns the appropriate device for PyTorch computations,
    
    dataset = load_wikitext() ## Loads the WikiText-2 dataset,

    train_dataset = clean_dataset(dataset["train"]) ## Cleans the training dataset by removing empty lines and unnecessary whitespace, preparing it for tokenization and model training.
    val_dataset = clean_dataset(dataset["validation"]) ## Cleans the validation dataset by removing empty lines and unnecessary whitespace, preparing it for tokenization and model evaluation.
    test_dataset= clean_dataset(dataset["test"]) ## Cleans the test dataset by removing empty lines and unnecessary whitespace, preparing it for tokenization and final evaluation of the trained model.

    tokenizer = get_tokenizer() ## Initializes a tokenizer (e.g., BERT tokenizer) that will be used to convert raw text into token IDs for model input and to decode model outputs back into human-readable text.

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

    # Tokenize test
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)
    test_sequences = create_fixed_length_sequences(
        tokenized_test,
        seq_len=256,
        stride=32
    )

    num_epochs = 4 # Set to 4 for quick testing

    if mode == "baseline": ## Runs the baseline training loop using a standard BERT denoising autoencoder architecture.

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


        train_loader = DataLoader( ## Creates a DataLoader for the training dataset, which will handle batching and shuffling of the data during training.
            train_data,
            batch_size=32,
            shuffle=True,
        )

        val_loader = DataLoader( ## Creates a DataLoader for the validation dataset, which will handle batching of the data during evaluation, without shuffling to maintain consistency.
            val_data,
            batch_size=32,
            shuffle=False,
        )

        print("\nRunning BASELINE training...\n")

        model = BertDenoiser().to(device) ## Initializes the BERT denoising autoencoder model and moves it to the appropriate device (CPU or MPS) for training.

        optimizer = torch.optim.AdamW( ## Sets up the AdamW optimizer with the model parameters, a learning rate of 3e-4, and a weight decay of 0.01 for regularization during training.
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )

        for epoch in range(num_epochs):

            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_one_epoch( ## Trains the model for one epoch by iterating over the training dataloader, applying masking, and computing the loss and accuracy on the masked tokens to update the model parameters.
                model,
                train_loader,
                optimizer,
                device,
            )

            val_loss, val_acc = evaluate(## Evaluates the model on the validation set by following a similar process as training but without gradient updates, to compute the average loss and accuracy on the masked tokens.
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
        
        T = 12
        mask_type = "span"
        mask_ratio = 0.10

        best_val_acc = 0.0

        train_data = TextInpaintingDataset( ## Creates a TextInpaintingDataset for the training data, which will apply the specified masking strategy.
            sequences=train_sequences,
            tokenizer=tokenizer,
            mask_type=mask_type,
            mask_ratio=mask_ratio,
            dynamic_masking=True,
        )

        val_data = TextInpaintingDataset( ## Creates a TextInpaintingDataset for the validation data, which will apply the specified masking strategy in a fixed manner for consistent evaluation.
            sequences=val_sequences,
            tokenizer=tokenizer,
            mask_type=mask_type,
            mask_ratio=mask_ratio,
            dynamic_masking=False,
        )

        train_loader = DataLoader( ## Creates a DataLoader for the training dataset, which will handle batching and shuffling of the data during training.
            train_data,
            batch_size=16,
            shuffle=True,
        )

        val_loader = DataLoader( ## Creates a DataLoader for the validation dataset, which will handle batching of the data during evaluation, without shuffling to maintain consistency.
            val_data,
            batch_size=16,
            shuffle=False,
        )

        model = DiffusionBert(T=T, conditioning_dropout=0.1).to(device) ## Initializes the DiffusionBert model with the specified number of diffusion steps (T) and conditioning dropout, and moves it to the appropriate device for training.

        diffusion_forward = DiscreteDiffusionForward(## Initializes the forward diffusion process with the specified number of steps (T) and the mask token ID from the tokenizer, and moves it to the appropriate device for training.
            T=T,
            mask_token_id=tokenizer.mask_token_id
        ).to(device)

        optimizer = torch.optim.AdamW( ## Sets up the AdamW optimizer with the model parameters and a learning rate of 3e-5 for training the diffusion model.
            model.parameters(),
            lr=3e-5,
        )
        print(train_loader.dataset[0].keys())
        for epoch in range(num_epochs):

            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_diffusion_epoch( ## Trains the diffusion model for one epoch by iterating over the training dataloader, sampling random timesteps, corrupting the target sequences according to the diffusion process, and computing the loss only on the masked positions to update the model parameters.
                model,
                train_loader,
                optimizer,
                diffusion_forward,
                tokenizer,
                device,
            )

            val_loss, val_acc = evaluate_diffusion( ## Evaluates the diffusion model on the validation set by following a similar process as training but without gradient updates, to compute the average loss and accuracy on the masked tokens.
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

            # SAVE BEST MODEL
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f"diffusion_{mask_type}_{mask_ratio}_T{T}_dropout_0.1.pt"
                )
                print("✅ Best model saved.")

        
    elif mode == "inference":

        def highlight_generated(original_ids, generated_ids, mask_positions, tokenizer): ## Highlights the generated tokens that were filled in for the masked positions by comparing the original token IDs, generated token IDs, and mask positions, and returns a string with the filled tokens highlighted in green.
            output_tokens = []

            for orig_id, gen_id, is_mask in zip( ##Iterates through the original token IDs, generated token IDs, and mask positions simultaneously to construct the output string with highlighted tokens.
                original_ids,
                generated_ids,
                mask_positions
            ):
                token = tokenizer.convert_ids_to_tokens(gen_id.item()) ## Converts the generated token ID back to its corresponding token string using the tokenizer.

                if is_mask:
                    # Green color for replaced tokens
                    token = f"\033[92m{token}\033[0m"

                output_tokens.append(token)

            return tokenizer.convert_tokens_to_string(output_tokens)
        
        print("\nRunning INFERENCE...\n")

        T = 12

        model = DiffusionBert( ## Initializes the DiffusionBert model with the specified number of diffusion steps (T) and conditioning dropout, and moves it to the appropriate device for inference.
            T=T,
            conditioning_dropout=0.1
        ).to(device)

        diffusion_forward = DiscreteDiffusionForward( ## Initializes the forward diffusion process with the specified number of steps (T) and the mask token ID from the tokenizer, and moves it to the appropriate device for inference.
            T=T,
            mask_token_id=tokenizer.mask_token_id
        ).to(device)

        model.load_state_dict( ## Loads the trained model weights from the specified file, mapping them to the appropriate device for inference.
            torch.load("diffusion_span_0.1_T12_dropout_0.1.pt", map_location=device)
        )

        model.eval()

        # Use validation dataset for testing
        val_data = TextInpaintingDataset( ## Creates a validation dataset for testing the inference performance of the diffusion model.
            sequences=val_sequences,
            tokenizer=tokenizer,
            mask_type="span",
            mask_ratio=0.10,
            dynamic_masking=False,
        )

        sample = val_data[0]
        ## Prepare input IDs and mask positions for the sample, moving them to the appropriate device for inference.
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        mask_positions = sample["mask_positions"].unsqueeze(0).to(device)

        from inference.reverse_diffusion import reverse_diffusion_sample

        generated = reverse_diffusion_sample( ## Runs the reverse diffusion sampling process using the trained model, forward diffusion process, tokenizer, input IDs, and mask positions to generate the inpainted token IDs for the masked positions.
            model,
            diffusion_forward,
            tokenizer,
            input_ids,
            mask_positions,
            T=T,
            temperature=0.8,
            top_k=20,
            device=device
        )

        original_text = tokenizer.decode( ## Decodes the original target token IDs back into a human-readable string using the tokenizer, skipping any special tokens in the process.
            sample["target_ids"],
            skip_special_tokens=True
        )

        masked_text = tokenizer.decode( ## Decodes the masked input token IDs back into a human-readable string using the tokenizer, without skipping special tokens to show the masked positions clearly.
            sample["input_ids"],
            skip_special_tokens=False
        )

        highlighted_text = highlight_generated( ## Highlights the generated tokens that were filled in for the masked positions by comparing the original token IDs, generated token IDs, and mask positions, and returns a string with the filled tokens highlighted in green.
            sample["target_ids"],
            generated[0].cpu(),
            sample["mask_positions"],
            tokenizer
        )

        print("\n" + "="*80)
        print("ORIGINAL:\n")
        print(original_text)

        print("\n" + "="*80)
        print("MASKED:\n")
        print(masked_text)

        print("\n" + "="*80)
        print("GENERATED (Highlighted Filled Tokens):\n")
        print(highlighted_text)
        print("="*80)


    elif mode == "test":

        print("\nRunning TEST evaluation...\n")

        best_val_acc = 0.0
        T = 12
        mask_ratio = 0.10   
        batch_size = 16

        # Create test dataset
        test_data = TextInpaintingDataset(
            sequences=test_sequences,
            tokenizer=tokenizer,
            mask_type="span",
            mask_ratio=mask_ratio,
            dynamic_masking=False,  # IMPORTANT
        )

        test_loader = DataLoader( ## Creates a DataLoader for the test dataset, which will handle batching of the data during evaluation, without shuffling to maintain consistency.
            test_data,
            batch_size=batch_size,
            shuffle=False,
        )

        # Load trained model
        model = DiffusionBert(
            T=T,
            conditioning_dropout=0.1,
        ).to(device)

        model.load_state_dict( ## Loads the trained model weights from the specified file, mapping them to the appropriate device for evaluation.
            torch.load("diffusion_span_0.1_T12_dropout_0.1.pt", map_location=device)
        )

        diffusion_forward = DiscreteDiffusionForward( ## Initializes the forward diffusion process with the specified number of steps (T) and the mask token ID from the tokenizer, and moves it to the appropriate device for evaluation.
            T=T,
            mask_token_id=tokenizer.mask_token_id,
        ).to(device)

        # Run evaluation
        test_loss, test_acc = evaluate_diffusion(
            model,
            test_loader,
            diffusion_forward,
            tokenizer,
            device,
        )

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # =========================
    # BLEU Evaluation
    # =========================

    print("\nComputing BLEU Score...\n")

    model.eval() ## Set the model to evaluation mode, which disables dropout and other training-specific behaviors, ensuring deterministic outputs during evaluation.

    total_bleu = 0
    total_rouge = 0
    num_samples = 0

    for batch_idx, batch in enumerate(test_loader):

        if batch_idx > 30:   # limit for speed
            break

        ## Prepare input IDs and mask positions for the batch, moving them to the appropriate device for evaluation.
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask_positions = batch["mask_positions"].to(device)

        ## Run reverse diffusion sampling to generate inpainted token IDs for the masked positions using the trained model, forward diffusion process, tokenizer, input IDs, and mask positions.
        generated = reverse_diffusion_sample(
            model=model,
            diffusion_forward=diffusion_forward,
            tokenizer=tokenizer,
            input_ids=input_ids,
            mask_positions=mask_positions,
            T=T,
            temperature=0.7,
            top_k=0,
            device=device
        )

        for i in range(input_ids.size(0)):

            ## Compute masked BLEU and ROUGE scores for each sample in the batch by comparing the target token IDs, generated token IDs, and mask positions, and accumulate the scores to compute the average at the end.
            bleu = compute_masked_bleu(
                target_ids[i].cpu(),
                generated[i].cpu(),
                mask_positions[i].cpu(),
                tokenizer
            )

            rouge = compute_masked_rouge_l(
                target_ids[i].cpu(),
                generated[i].cpu(),
                mask_positions[i].cpu(),
                tokenizer
            )

            total_bleu += bleu
            total_rouge += rouge
            num_samples += 1

    avg_bleu = total_bleu / num_samples
    avg_rouge = total_rouge / num_samples

    print(f"Masked BLEU Score: {avg_bleu:.4f}")
    print(f"Masked ROUGE-L Score: {avg_rouge:.4f}")