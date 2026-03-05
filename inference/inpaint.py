## Runs the full inference pipeline for text inpainting using the trained diffusion model.

import torch
from data.masking import apply_masking
from inference.reverse_diffusion import reverse_diffusion_sample


def inpaint_text( ## Main function to perform text inpainting. It takes the input text, applies masking, runs the reverse diffusion process to generate inpainted text, and decodes the output back to a string.
    text,
    model,
    diffusion_forward,
    tokenizer,
    T=12,
    mask_type="span",
    mask_ratio=0.10,
    temperature=0.8,
    top_k=20,
    device="cpu" 
):
    model.eval()

    # 1️⃣ Tokenize
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    input_ids = tokens["input_ids"].to(device)

    # 2️⃣ Apply SAME masking logic used during training
    masked_input, target_ids, mask_positions = apply_masking(
        input_ids=input_ids.squeeze(0),
        mask_token_id=tokenizer.mask_token_id,
        mask_type=mask_type,
        mask_ratio=mask_ratio,
    )

    masked_input = masked_input.unsqueeze(0).to(device)
    mask_positions = mask_positions.unsqueeze(0).to(device)

    # 3️⃣ Run reverse diffusion
    generated_ids = reverse_diffusion_sample(
        model=model,
        diffusion_forward=diffusion_forward,
        tokenizer=tokenizer,
        input_ids=masked_input,
        mask_positions=mask_positions,
        T=T,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    # 4️⃣ Decode
    generated_text = tokenizer.decode(
        generated_ids.squeeze(0),
        skip_special_tokens=True
    )

    return generated_text