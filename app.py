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

model = DiffusionBert(
    T=T,
    conditioning_dropout=0.1
).to(device)

model.load_state_dict(
    torch.load("diffusion_span_0.1_T12_dropout_0.1.pt", map_location=device)
)

model.eval()

diffusion_forward = DiscreteDiffusionForward(
    T=T,
    mask_token_id=tokenizer.mask_token_id
).to(device)


# =============================
# Highlight Function (HTML)
# =============================

def highlight_tokens(masked_ids, generated_ids, mask_positions):
    tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
    masked_flags = mask_positions.tolist()

    final_tokens = []

    for token, is_mask in zip(tokens, masked_flags):
        if token in tokenizer.all_special_tokens:
            continue

        if is_mask:
            token = f"<span style='color:green; font-weight:bold'>{token}</span>"

        final_tokens.append(token)

    return tokenizer.convert_tokens_to_string(final_tokens)


# =============================
# Main Inpainting Function
# =============================

def inpaint(text, temperature, top_k):

    encoded = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=False,
    max_length=256
    )

    input_ids = encoded["input_ids"][0]

    input_ids = input_ids.to(device)

    masked_input, _, mask_positions = apply_masking(
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

    generated = reverse_diffusion_sample(
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
    tokens = tokenizer.convert_ids_to_tokens(
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

    masked_text = tokenizer.convert_tokens_to_string(tokens)

    highlighted_text = highlight_tokens(
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
    fn=inpaint,
    inputs=[
        gr.Textbox(label="Input Text", lines=8),
        gr.Slider(0.5, 1.5, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(0, 50, value=20, step=1, label="Top-K"),
    ],
    outputs=[
        gr.HTML(label="Output")
    ],
    title="Diffusion Text Inpainting",
    description="Paste text → model masks spans → diffusion fills them → filled tokens shown in green."
)

if __name__ == "__main__":
    demo.launch()