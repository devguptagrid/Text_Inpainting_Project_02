import random
import torch


def random_token_mask(
    input_ids,
    mask_token_id,
    mask_ratio=0.25,
):
    
    ##Randomly masks individual tokens (non-contiguous).
    

    seq_len = len(input_ids) ## number of tokens in the input sequence
    num_to_mask = int(seq_len * mask_ratio) ## number of tokens to mask based on the specified ratio

    masked_input = input_ids.copy()  ## create a copy of the input token IDs to modify for masking
    mask_positions = [False] * seq_len ## initialize a list to track which positions are masked (False means not masked, True means masked)

    mask_indices = random.sample(range(seq_len), num_to_mask) ## randomly select indices to mask based on the number of tokens to mask

    for idx in mask_indices: ## iterate over the selected indices and apply masking
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
):

    ##Applies contiguous span masking.


    seq_len = len(input_ids)
    num_to_mask = int(seq_len * mask_ratio)

    masked_input = input_ids.copy()
    mask_positions = [False] * seq_len

    total_masked = 0

    while total_masked < num_to_mask:
        span_length = random.randint(min_span_length, max_span_length) ## randomly determine the length of the span to mask within the specified min and max span lengths
        start_idx = random.randint(0, seq_len - span_length) ## randomly select a starting index for the span, ensuring that the span fits within the sequence length

        for i in range(start_idx, start_idx + span_length):  ## iterate over the indices of the selected span and apply masking
            if not mask_positions[i]: ## only mask if the position is not already masked (to avoid double masking)
                masked_input[i] = mask_token_id ## replace the token ID at the current index with the mask token ID
                mask_positions[i] = True ## mark the position as masked in the mask_positions list
                total_masked += 1 ## increment the total masked count, and if it reaches the number of tokens to mask, break out of the loop to stop masking further spans

                if total_masked >= num_to_mask:
                    break

    return (
        torch.tensor(masked_input, dtype=torch.long),
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(mask_positions, dtype=torch.bool),
    )


def apply_masking(
    input_ids,
    mask_token_id,
    mask_type="span",
    mask_ratio=0.25,
):
    

    if mask_type == "span": ## if the specified mask type is "span", call the span_mask_sequence function to apply contiguous span masking to the input token IDs
        return span_mask_sequence(
            input_ids,
            mask_token_id,
            mask_ratio=mask_ratio,
        )

    elif mask_type == "random":
        return random_token_mask(
            input_ids,
            mask_token_id,
            mask_ratio=mask_ratio,
        )

    else:
        raise ValueError("mask_type must be 'span' or 'random'")