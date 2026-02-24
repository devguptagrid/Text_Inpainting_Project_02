import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_diffusion_epoch(
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

    for batch in tqdm(dataloader, desc="Training Diffusion"):

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        batch_size = input_ids.size(0)

        # Sample random timesteps
        t = diffusion_forward.sample_timestep(batch_size, device)
        t_embed = t - 1  # convert to 0-index

        # Generate x_t
        x_t = diffusion_forward.corrupt(input_ids, t)

        attention_mask = torch.ones_like(x_t, dtype=torch.bool)

        # Forward pass
        logits = model(x_t, t_embed, attention_mask)

        # Compute loss only on masked positions
        mask = (x_t == tokenizer.mask_token_id)

        loss = F.cross_entropy(
            logits[mask],
            target_ids[mask]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == target_ids[mask]).sum().item()

        total_correct += correct
        total_masked += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked

    return avg_loss, accuracy

def evaluate_diffusion(
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
        for batch in tqdm(dataloader, desc="Validation Diffusion"):

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            batch_size = input_ids.size(0)

            t = diffusion_forward.sample_timestep(batch_size, device)
            t_embed = t - 1

            x_t = diffusion_forward.corrupt(input_ids, t)

            attention_mask = torch.ones_like(x_t, dtype=torch.bool)

            logits = model(x_t, t_embed, attention_mask)

            mask = (x_t == tokenizer.mask_token_id)

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