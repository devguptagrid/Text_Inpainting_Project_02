import torch


def masked_accuracy(
    logits,
    target_ids,
    mask_positions,
):
    """
    Computes accuracy only on masked tokens.

    Args:
        logits: (B, seq_len, vocab_size)
        target_ids: (B, seq_len)
        mask_positions: (B, seq_len) bool

    Returns:
        accuracy (float)
    """

    # Get predicted token IDs
    predictions = torch.argmax(logits, dim=-1)

    # Select masked positions
    masked_preds = predictions[mask_positions]
    masked_targets = target_ids[mask_positions]

    correct = (masked_preds == masked_targets).sum().item()
    total = masked_targets.numel()

    if total == 0:
        return 0.0

    return correct / total