## Dataset

We use WikiText-2 (raw version) from HuggingFace.

Original dataset sizes:
- Train: 36,718
- Validation: 3,760
- Test: 4,358

After cleaning (removing empty lines and very short sequences<10 character>):
- Train: 23,547
- Validation: 2,454
- Test: 2,850

Cleaning removes empty lines, section headers, and very short fragments to improve training stability and quality.





Baseline Development & Improvements
Initial Baseline: Transformer From Scratch

We first implemented a masked language modeling (MLM) baseline using:

Custom Transformer Encoder

Dynamic span masking

Masked cross-entropy loss

Masked-token accuracy metric

Train/Validation split

Experiments Conducted

We experimented with:

Masking type: span masking

Mask ratios: 0.25 → reduced to 0.10

Increased epochs: 5+

Increased encoder capacity:

Hidden size: 256 → 384

Layers: 4 → 6

Heads: 4 → 6

Fixed validation masking for stable evaluation

Observation

Although training loss consistently decreased:

Masked-token accuracy remained around 5–6%

Increasing model size and training epochs did not significantly improve top-1 accuracy

Reducing mask ratio to 0.10 also did not improve results

Reason

Training a Transformer from scratch on WikiText-2 (~2M tokens) is insufficient for learning strong language representations. The vocabulary size (~30k tokens) and limited dataset size make exact token prediction extremely difficult without large-scale pretraining.

Final Baseline Strategy: Fine-tuning Pretrained BERT

To build a strong and realistic baseline before diffusion modeling, we switched to:

bert-base-uncased as pretrained encoder backbone

Added linear output layer for token prediction

Fine-tuned on span masking objective

Why This Change?

Provides strong pretrained language representations

Enables meaningful masked-token prediction

Aligns with modern NLP practice

Establishes a solid baseline before implementing D3PM

This pretrained fine-tuned BERT model serves as the final baseline prior to diffusion-based inpainting experiments.