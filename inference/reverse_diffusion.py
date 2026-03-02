import torch
import torch.nn.functional as F


def sample_with_temperature_topk(logits, temperature=1.0, top_k=0):
    logits = logits / temperature ##Temperature controls randomness:

    if top_k > 0:
        topk_vals, topk_indices = torch.topk(logits, top_k) ## Select the top_k highest logits and their corresponding token indices, topk_vals = values of the top_k logits, topk_indices = token indices of the top_k logits
        probs = F.softmax(topk_vals, dim=-1) ## Convert the top_k logits to probabilities using softmax, 
        next_token = torch.multinomial(probs, 1) ## Sample one token index from the top_k probabilities, next_token = index of the sampled token within the top_k candidates
        return topk_indices.gather(-1, next_token) ##
    else:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)


def reverse_diffusion_sample(
    model,
    diffusion_forward,
    tokenizer,
    input_ids,
    mask_positions,
    T,
    temperature=1.0,
    top_k=0,
    device="cpu"
):
    model.eval() ## Set the model to evaluation mode, which disables dropout and other training-specific behaviors, ensuring deterministic outputs during sampling.

    x_t = input_ids.clone().to(device) ## Create a copy of the input_ids tensor and move it to the specified device (CPU or GPU) for processing.

    with torch.no_grad(): ##

        for t in reversed(range(1, T + 1)):

            t_tensor = torch.full( ## Create a tensor filled with the current timestep t for each sample in the batch, which will be used as input to the model to condition the generation process on the current diffusion step.
                (x_t.size(0),),
                t,
                device=device,
                dtype=torch.long
            )

            t_embed = t_tensor - 1

            attention_mask = torch.ones_like(x_t, dtype=torch.bool)

            logits = model(
                x_t,
                t_embed,
                mask_positions,
                attention_mask
            )

            logits_masked = logits[mask_positions]

            next_tokens = sample_with_temperature_topk(
                logits_masked,
                temperature=temperature,
                top_k=top_k
            ).squeeze(-1)

            x_t[mask_positions] = next_tokens

    return x_t