import torch
import torch.nn.functional as F


def masked_cross_entropy_loss(
    logits,
    target_ids,
    mask_positions,
):
    """
    Computes cross-entropy loss only on masked tokens.

    Args:
        logits: (B, seq_len, vocab_size)
        target_ids: (B, seq_len)
        mask_positions: (B, seq_len) bool

    Returns:
        scalar loss
    """

    # Flatten tensors
    B, seq_len, vocab_size = logits.shape

    logits = logits.view(-1, vocab_size) ## reshape logits to (B*seq_len, vocab_size) for loss computation
    target_ids = target_ids.view(-1) ## reshape target_ids to (B*seq_len) to align with the flattened logits
    mask_positions = mask_positions.view(-1) ## reshape mask_positions to (B*seq_len) to identify which positions are masked in the flattened format

    # Select only masked positions
    logits_masked = logits[mask_positions]
    targets_masked = target_ids[mask_positions]

    loss = F.cross_entropy(logits_masked, targets_masked) ## compute cross-entropy loss between the predicted logits for masked positions and the true target token IDs for those positions, loss = -log(softmax(logits)[correct_token])

    return loss