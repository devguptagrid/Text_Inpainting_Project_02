##Training loop for the diffusion model including timestep sampling, corruption, and denoising loss computation.

import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_diffusion_epoch( ## Trains the diffusion model for one epoch by iterating over the training dataloader, sampling random timesteps, corrupting the target sequences according to the diffusion process, and computing the loss only on the masked positions to update the model parameters.
    model,
    dataloader,
    optimizer,
    diffusion_forward,
    tokenizer,
    device,
):
    model.train()

    total_loss = 0
    total_correct = 0
    total_masked = 0

    accumulation_steps = 2 ## Number of steps to accumulate gradients before updating model parameters, helps with memory efficiency by simulating a larger batch size without increasing memory usage.
    optimizer.zero_grad() ## Clear gradients at the start of the epoch to ensure that gradients from previous epochs do not affect the current training process.

    for step, batch in enumerate(tqdm(dataloader, desc="Training Diffusion")):

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        batch_size = input_ids.size(0) 

        # Sample random timesteps
        t = diffusion_forward.sample_timestep(batch_size, device)
        t_embed = t - 1  # convert to 0-index

        # Corrupt input according to diffusion process 
        span_mask = batch["mask_positions"].to(device)
        x_t = diffusion_forward.corrupt(target_ids, t, span_mask)
        mask_positions = span_mask
        
        # Attention mask (all ones since we want to attend to all tokens)
        attention_mask = torch.ones_like(x_t, dtype=torch.bool) 

        # Forward pass
        logits = model(x_t, t_embed, mask_positions, attention_mask)

        # Compute loss only on masked positions
        mask = span_mask

        loss = F.cross_entropy(
            logits[mask],
            target_ids[mask]
        )

        # Divide loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Update only every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute accuracy and loss on masked positions 
        total_loss += loss.item() * accumulation_steps

        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == target_ids[mask]).sum().item()

        total_correct += correct
        total_masked += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked

    return avg_loss, accuracy

def evaluate_diffusion( ## Evaluates the diffusion model on the validation set by following a similar process as training but without gradient updates, to compute the average loss and accuracy on the masked tokens.
    model,
    dataloader,
    diffusion_forward,
    tokenizer,
    device,
):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_masked = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation Diffusion"):

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            batch_size = input_ids.size(0)

            # Sample random timesteps
            t = diffusion_forward.sample_timestep(batch_size, device)
            t_embed = t - 1
            # Corrupt input according to diffusion process
            span_mask = batch["mask_positions"].to(device)

            #Corrupt input according to diffusion process 
            x_t = diffusion_forward.corrupt(target_ids, t, span_mask)

            # Attention mask (all ones since we want to attend to all tokens)
            attention_mask = torch.ones_like(x_t, dtype=torch.bool)

            # Forward pass
            logits = model(x_t, t_embed, span_mask, attention_mask)

            mask = span_mask

            # Compute loss and accuracy only on masked positions
            loss = F.cross_entropy(
                logits[mask],
                target_ids[mask]
            )

            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == target_ids[mask]).sum().item()

            total_correct += correct
            total_masked += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked

    return avg_loss, accuracy