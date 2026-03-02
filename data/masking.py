import random
import torch


def random_token_mask(
    input_ids,
    mask_token_id,
    mask_ratio=0.25,
    special_token_ids=None,
):
    
    ##Randomly masks individual tokens (non-contiguous).
    

    seq_len = len(input_ids) ## number of tokens in the input sequence
    num_to_mask = int(seq_len * mask_ratio) ## number of tokens to mask based on the specified ratio

    masked_input = input_ids.clone()  ## create a copy of the input token IDs to modify for masking
    mask_positions = [False] * seq_len ## initialize a list to track which positions are masked (False means not masked, True means masked)

    mask_indices = random.sample(range(seq_len), num_to_mask) ## randomly select indices to mask based on the number of tokens to mask

    for idx in mask_indices: ## iterate over the selected indices and apply masking
        if input_ids[idx].item() in special_token_ids:
            continue
        masked_input[idx] = mask_token_id ## replace the token ID at the selected index with the mask token ID
        mask_positions[idx] = True ## mark the position as masked in the mask_positions list

    return (
        torch.tensor(masked_input, dtype=torch.long),
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(mask_positions, dtype=torch.bool),
    )



def span_mask_sequence(
    input_ids,
    mask_token_id,
    mask_ratio=0.25,
    min_span_length=3,
    max_span_length=10,
    special_token_ids=None,
):

    seq_len = len(input_ids)
    num_to_mask = int(seq_len * mask_ratio)

    masked_input = input_ids.clone()
    mask_positions = torch.zeros(seq_len, dtype=torch.bool)

    total_masked = 0

    # Default special tokens protection
    if special_token_ids is None:
        special_token_ids = set()

    while total_masked < num_to_mask:

        span_length = random.randint(min_span_length, max_span_length)
        start_idx = random.randint(0, seq_len - span_length)

        for i in range(start_idx, start_idx + span_length):

            # 🚫 Do NOT mask special tokens
            if input_ids[i].item() in special_token_ids:
                continue

            # 🚫 Skip if already masked
            if mask_positions[i]:
                continue

            masked_input[i] = mask_token_id
            mask_positions[i] = True
            total_masked += 1

            if total_masked >= num_to_mask:
                break

    return masked_input, input_ids.clone(), mask_positions


def apply_masking(
    input_ids,
    mask_token_id,
    mask_type="span",
    mask_ratio=0.25,
    special_token_ids=None,
):

    if mask_type == "span":
        return span_mask_sequence(
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            mask_ratio=mask_ratio,
            special_token_ids=special_token_ids,
        )

    elif mask_type == "random":
        return random_token_mask(
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            mask_ratio=mask_ratio,
            special_token_ids=special_token_ids,
        )

    else:
        raise ValueError("mask_type must be 'span' or 'random'")