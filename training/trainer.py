import torch
from tqdm import tqdm ## progress bar for training loop
from training.loss import masked_cross_entropy_loss
from evaluation.metrics import masked_accuracy

def train_one_epoch(model, dataloader, optimizer, device): ##function for input model as TransformerDenoiser
    model.train()

    total_loss = 0
    total_acc=0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask_positions = batch["mask_positions"].to(device)

        optimizer.zero_grad() ## clears old gradients from the last step before computing new gradients for the current batch, preventing gradient accumulation across batches which can lead to incorrect updates and increased memory usage.

        logits = model(input_ids) ## forward pass through the model to get predicted logits for each token in the input sequence
        loss = masked_cross_entropy_loss( ## computes the loss by comparing the predicted logits with the true target token IDs, but only for the positions that were masked (where mask_positions is True), ensuring that the model is only penalized for incorrect predictions on the masked tokens and not on the unmasked tokens.
            logits,
            target_ids,
            mask_positions,
        )

        loss.backward() ##compte gradienyts wrt loss
        optimizer.step() ## update model parameters based on computed gradients
        acc = masked_accuracy( ## computes the accuracy of the model's predictions for the masked tokens by comparing the predicted token IDs (obtained by taking the argmax of the logits) with the true target token IDs, but only for the positions that were masked, giving a measure of how well the model is reconstructing the masked tokens.
            logits,
            target_ids,
            mask_positions,
        )
        total_loss += loss.item() ## accumulate loss for the epoch to compute average loss later
        total_acc += acc ## accumulate accuracy for the epoch to compute average accuracy later
        progress_bar.set_postfix(loss=loss.item(), acc=acc) ## update the progress bar to show the current loss and accuracy for the batch
    avg_loss = total_loss / len(dataloader) ##give mean loss across batches
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc

