## Used to create the Gradio UI

import torch
import gradio as gr

from models.diffusion_model import DiffusionBert
from diffusion.forward_process import DiscreteDiffusionForward
from inference.reverse_diffusion import reverse_diffusion_sample
from data.masking import apply_masking
from data.preprocessing import get_tokenizer
from utils.device import get_device

# =============================
# Load Model Once (Global)
# =============================

device = get_device()
tokenizer = get_tokenizer()

T = 12
mask_ratio = 0.10

model = DiffusionBert(## Initializes the diffusion model with the specified parameters, including the number of diffusion steps (T), the mask token ID from the tokenizer, and a conditioning dropout rate, and moves it to the appropriate device for inference.
    T=T,
    conditioning_dropout=0.1
).to(device)

model.load_state_dict( ## Loads the trained model weights from the specified file, mapping them to the appropriate device for inference.
    torch.load("diffusion_span_0.1_T12_dropout_0.1.pt", map_location=device)
)

model.eval()

diffusion_forward = DiscreteDiffusionForward( ## Initializes the forward diffusion process with the specified number of steps (T) and the mask token ID from the tokenizer, and moves it to the appropriate device for inference.
    T=T,
    mask_token_id=tokenizer.mask_token_id
).to(device)


# =============================
# Highlight Function (HTML)
# =============================

def highlight_tokens(masked_ids, generated_ids, mask_positions):
    tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist()) ## Convert the generated token IDs to their corresponding token strings using the tokenizer's vocabulary.
    masked_flags = mask_positions.tolist() ## Convert the mask positions tensor to a list of boolean flags indicating which tokens were masked (True for masked, False for unmasked).

    final_tokens = []

    for token, is_mask in zip(tokens, masked_flags):
        if token in tokenizer.all_special_tokens: ## If the token is a special token (like [CLS], [SEP], [PAD]), we skip highlighting and just add it to the final tokens list without modification.
            continue

        if is_mask: ## If the token was masked (is_mask is True), we wrap it in an HTML span with green color and bold font to highlight it as a generated token.
            token = f"<span style='color:green; font-weight:bold'>{token}</span>"

        final_tokens.append(token) ## Add the token (highlighted or not) to the final tokens list.

    return tokenizer.convert_tokens_to_string(final_tokens)


# =============================
# Main Inpainting Function
# =============================

def inpaint(text, temperature, top_k): ## Main function to perform text inpainting. It takes the input text, applies masking, runs the reverse diffusion process to generate inpainted text, and decodes the output back to a string.

    encoded = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=False,
    max_length=256
    )

    input_ids = encoded["input_ids"][0] ## Extract the input token IDs from the tokenized output, which will be used as input for the masking and generation process.

    input_ids = input_ids.to(device) ## Move the input token IDs to the appropriate device (CPU or GPU) for processing during masking and generation.

    masked_input, _, mask_positions = apply_masking( ## Apply the same masking logic used during training to the input token IDs, which will create a masked version of the input and identify the positions that were masked.
        input_ids=input_ids,
        mask_token_id=tokenizer.mask_token_id,
        mask_type="span",
        mask_ratio=mask_ratio,
        special_token_ids={
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        }
    )

    masked_input = masked_input.unsqueeze(0).to(device)
    mask_positions = mask_positions.unsqueeze(0).to(device)

    generated = reverse_diffusion_sample( ## Run reverse diffusion sampling to generate inpainted token IDs for the masked positions using the trained model, forward diffusion process, tokenizer, input IDs, and mask positions.
        model=model,
        diffusion_forward=diffusion_forward,
        tokenizer=tokenizer,
        input_ids=masked_input,
        mask_positions=mask_positions,
        T=T,
        temperature=temperature,
        top_k=int(top_k),
        device=device
    )

    original_text = text
    tokens = tokenizer.convert_ids_to_tokens( ## Convert the original input token IDs to their corresponding token strings using the tokenizer's vocabulary, which will be used to reconstruct the masked text for display.
    masked_input.squeeze(0).tolist()
    )

    # Remove only CLS, SEP, PAD (keep [MASK])
    tokens = [
        tok for tok in tokens
        if tok not in [
            tokenizer.cls_token,
            tokenizer.sep_token,
            tokenizer.pad_token
        ]
    ]

    masked_text = tokenizer.convert_tokens_to_string(tokens) ## Convert the list of tokens (with special tokens removed) back to a string using the tokenizer's vocabulary, which will represent the masked version of the original text for display.

    highlighted_text = highlight_tokens( ## Call the highlight_tokens function to generate an HTML string with the generated tokens highlighted in green, by comparing the masked input token IDs, generated token IDs, and mask positions.
        masked_input.squeeze(0).cpu(),
        generated.squeeze(0).cpu(),
        mask_positions.squeeze(0).cpu()
    )

    boxed_output = f"""
    <div style="display:flex; flex-direction:column; gap:16px;">


    <div style="
        border:1px solid #444;
        border-radius:8px;
        padding:12px;
        max-height:250px;
        overflow-y:auto;
        background-color:#1e1e1e;
    ">
    <b>Masked</b><br><br>
    {masked_text}
    </div>

    <div style="
        border:1px solid #444;
        border-radius:8px;
        padding:12px;
        max-height:250px;
        overflow-y:auto;
        background-color:#1e1e1e;
    ">
    <b>Generated (Highlighted)</b><br><br>
    {highlighted_text}
    </div>

    </div>
    """

    return boxed_output


# =============================
# Gradio UI
# =============================

demo = gr.Interface(
    fn=inpaint,## Set the function to be called when the user interacts with the UI, which is the inpaint function defined above that performs text inpainting and returns an HTML string with the results.
    inputs=[ ## Define the input components for the Gradio UI, which include a textbox for the input text, and sliders for adjusting the temperature and top-k parameters for generation.
        gr.Textbox(label="Input Text", lines=8),
        gr.Slider(0.5, 1.5, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(0, 50, value=20, step=1, label="Top-K"),
    ],
    outputs=[ ## Define the output component for the Gradio UI, which is an HTML component that will display the masked and generated text with appropriate formatting and highlighting.
        gr.HTML(label="Output")
    ],
    title="Diffusion Text Inpainting",
    description="Paste text → model masks spans → diffusion fills them → filled tokens shown in green."
)

if __name__ == "__main__":
    demo.launch()