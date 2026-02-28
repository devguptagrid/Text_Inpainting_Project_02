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





## Baseline Development & Improvements
### Initial Baseline: Transformer From Scratch

We first implemented a masked language modeling (MLM) baseline using:

Custom Transformer Encoder

Dynamic span masking

Masked cross-entropy loss

Masked-token accuracy metric

Train/Validation split

### Experiments Conducted

We experimented with:

Masking type: span masking

Mask ratios: 0.25 → reduced to 0.10

Increased epochs: 5+

Increased encoder capacity:

Hidden size: 256 → 384

Layers: 4 → 6

Heads: 4 → 6

Fixed validation masking for stable evaluation

### Observation

Although training loss consistently decreased:

Masked-token accuracy remained around 5–6%

Reducing mask ratio to 0.10 also did not improve results

### Reason

Training a Transformer from scratch on WikiText-2 (~2M tokens) is insufficient for learning strong language representations. The vocabulary size (~30k tokens) and limited dataset size make exact token prediction extremely difficult without large-scale pretraining.

### Initial Baseline Model (Built From Scratch)

Before switching to pretrained BERT, we implemented a custom Transformer model from scratch using PyTorch.

Final Baseline Strategy: Fine-tuning Pretrained BERT

To build a strong and realistic baseline before diffusion modeling, we switched to:

BertForMaskedLM model class and used bert-based-uncased model ( uncased means text is lowercase before tokenization)

Fine-tuned on span masking objective

### Why This Change?

Provides strong pretrained language representations

Enables meaningful masked-token prediction

Aligns with modern NLP practice

Establishes a solid baseline before implementing D3PM

This pretrained fine-tuned BERT model serves as the final baseline prior to diffusion-based inpainting experiments.


## Technical Implementation Details
1️⃣ Dataset Loading

Library: datasets (HuggingFace)

Function Used: load_dataset("wikitext", "wikitext-2-raw-v1")


2️⃣ Text Cleaning

Method: Dataset filtering

Function Used: .filter()

3️⃣ Tokenization

Library: transformers

Tokenizer Used: AutoTokenizer.from_pretrained("bert-base-uncased")

Purpose: Convert raw text into token IDs using WordPiece vocabulary (~30k tokens).

4️⃣ Fixed-Length Chunking (256 Tokens)

Method: Manual sequence concatenation + slicing

Libraries Used: itertools.chain, Python list slicing

5️⃣ Span Masking / Random Masking

Custom Implementation: apply_masking()

Tech Used: Python random sampling + token replacement with [MASK] ID

Random token masking

Contiguous span masking

6️⃣ PyTorch Dataset & DataLoader

Library: torch.utils.data

Classes Used: Dataset, DataLoader

Purpose:

Dynamic masking during training

Fixed masking during validation

Batch processing

7️⃣ Initial Model (From Scratch)

Built using PyTorch modules:

nn.Embedding – Token embedding layer

nn.Embedding – Positional embedding layer

nn.TransformerEncoderLayer – Self-attention + feed-forward block

nn.TransformerEncoder – Stacked Transformer layers

nn.Linear – Output projection to vocabulary size

Purpose: Learn masked token reconstruction from scratch.

8️⃣ Loss Function

Library: torch.nn.functional

Function Used: F.cross_entropy()

9️⃣ Optimizer

Library: torch.optim

Optimizer Used: AdamW

Purpose: Update model parameters using backpropagation.

🔟 Pretrained Model Fine-Tuning

Library: transformers

Model Used: BertModel.from_pretrained("bert-base-uncased")

Added:

Custom nn.Linear output head

Fine-tuning:

All BERT parameters updated using optimizer = AdamW(model.parameters())

Purpose: Leverage pretrained language representations for stronger masked-token prediction.



## Why Do You See Two “Total sequences created”?

You see:

Total sequences created: 9180
Total sequences created: 951

These correspond to:

First → Training Set

9180 sequences

Second → Validation Set

951 sequences

Remember in main.py we did:

train_dataset = dataset["train"]

val_dataset = dataset["validation"]

Then we tokenized and chunked both separately.

So:

Split	Raw Rows (after cleaning)	256-token sequences

Train	~23k lines	which is around 2000000 tokens 

9180 unique sequences

Validation	~2.4k lines	

951 unique sequences

So nothing is being duplicated — it’s just:

Train sequences
Validation sequences

🔹 2️⃣ What Is 287 During Training?

You see:

Training: 100%| ... | 287/287

That means:

👉 287 batches per epoch.

Why 287?

Because:

Train sequences = 9180

Batch size = 32

So:

9180
÷
32
≈
287


That’s exactly what you’re seeing.



## 📊 Baseline Masking Experiments Results after 3 epochs for 9k sequences batch size 32

| Mask Type | Ratio | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|------------|--------|----------------|---------------------|------------|-----------------|
| Random     | 0.10   | 57.83          | 57.36               | 2.1694     | 2.2301          |
| Random     | 0.25   | 53.07          | 51.85               | 2.4602     | 2.6114          |
| Random     | 0.40   | 45.01          | 43.90               | 3.0190     | 3.2059          |
| Span       | 0.10   | 20.45          | 21.23               | 5.1082     | 5.1766          |
| Span       | 0.25   | 19.44          | 19.94               | 5.0969     | 5.2521          |
| Span       | 0.40   | 17.51          | 17.91               | 5.2213     | 5.4480          |

## Baseline model with 73k sequences with batch size of 32 without gradient accumulation for span masking and mask ratio of 0.25 after 3 epochs
Train Loss: 4.5502

Train Accuracy: 0.2179

Validation Loss: 5.4015

Validation Accuracy: 0.1989


## Diffusion model results

### after 6 epochs for batch size 16 , span masking and 0.25 mask ratio for 9k sequences
Train Loss: 5.0208
Train Accuracy: 0.2280
Validation Loss: 4.9633
Validation Accuracy: 0.2439


## Gradient Accumulation (Effective Batch Size 32 on MPS)

During diffusion training on Apple MPS, using batch_size=32 caused significant slowdown due to memory pressure from BERT-base and multi-step diffusion.

To maintain stable training while respecting project requirements (32–64 batch size), we used:

batch_size = 16
gradient_accumulation_steps = 2

This results in an effective batch size of 32, because gradients from two mini-batches of 16 are accumulated before performing a single optimizer update.

### Why This Is Valid

Gradients are additive. Accumulating gradients across multiple smaller batches and updating once is mathematically equivalent to training with a larger batch.

## after 6 epochs for batch size 16 and effective batch size 32 by gradient accumulation, span masking and 0.25 mask ratio for 73k sequences without mask conditioning
Train Loss: 4.5649
Train Accuracy: 0.2608
Validation Loss: 4.9749
Validation Accuracy: 0.2458


## in


## Big Picture Strategy

### Phase 1:

Large-data baseline

Large-data diffusion (implicit masking)

### Phase 2:

Add explicit mask conditioning

Compare improvement

### Phase 3:

Span vs random under diffusion

Ratio 0.1 / 0.25 / 0.4

Conditioning dropout

### Phase 4:

Inference experiments (temperature, top-k)

UI






## 🚀 Project Progression & Experiments

### 1️⃣ Baseline Model (From Scratch)
- Built a Transformer encoder model from scratch.
- Trained on ~9k sequences.
- Batch size: 32
- Masking: Span & Random
- Result: **5–6% masked-token accuracy**
- Conclusion: Model capacity and/or data scale insufficient.

---

### 2️⃣ Pretrained BERT Fine-Tuning Baseline
- Used `bert-base-uncased` (`BertForMaskedLM`).
- Fine-tuned for:
  - Mask types: Span & Random
  - Mask ratios: 10%, 25%, 40%
- Training:
  - 3 epochs
  - 9k sequences
  - Batch size: 32
- Observed accuracy trends across masking strategies.

| Mask Type | Ratio | Train Accuracy% | Validation Accuracy% | Train Loss | Validation Loss |
|------------|--------|----------------|---------------------|------------|-----------------|
| Random     | 0.10   | 57.83          | 57.36               | 2.1694     | 2.2301          |
| Random     | 0.25   | 53.07          | 51.85               | 2.4602     | 2.6114          |
| Random     | 0.40   | 45.01          | 43.90               | 3.0190     | 3.2059          |
| Span       | 0.10   | 20.45          | 21.23               | 5.1082     | 5.1766          |
| Span       | 0.25   | 19.44          | 19.94               | 5.0969     | 5.2521          |
| Span       | 0.40   | 17.51          | 17.91               | 5.2213     | 5.4480          |

---

### 3️⃣ Initial Diffusion Model
- Implemented Discrete Diffusion (D3PM-style).
- Training:
  - 6 epochs
  - Batch size: 16
  - Span masking
  - Mask ratio: 0.25
  - 9k sequences
- Goal: Compare diffusion vs single-step MLM baseline.

Train Loss: 5.0208

Train Accuracy: 22.80%

Validation Loss: 4.9633

Validation Accuracy: 24.39%

---

### 4️⃣ Increased Dataset Size (Sliding Window)
- Switched from unique chunking to sliding-window chunking.
- Increased sequences from ~9k → ~73k.
- Improved training diversity and coverage.

---

### 5️⃣ Baseline Model on 73k Sequences
- Span masking
- Mask ratio: 0.25
- Batch size: 32
- 3 epochs
- No gradient accumulation
- Evaluated scaling effect of larger dataset.

Train Loss: 4.5502

Train Accuracy: 21.79%

Validation Loss: 5.4015

Validation Accuracy: 19.89%

---

### 6️⃣ Gradient Accumulation Introduced
- Enabled effective larger batch sizes.
- Reduced GPU/MPS memory pressure.
- Improved training stability.

---

### 7️⃣ Diffusion Model (No Mask Conditioning)
- 6 epochs
- Batch size: 16
- Effective batch size: 32 (via gradient accumulation)
- Span masking
- Mask ratio: 0.25
- 73k sequences
- No explicit mask conditioning
- Evaluated pure diffusion denoising.

Train Loss: 4.5649

Train Accuracy: 26.08%

Validation Loss: 4.9749

Validation Accuracy: 24.58%

---

### 8️⃣ Diffusion Model (With Mask Conditioning)
- Same setup as above, but:
  - Added explicit mask conditioning
- 6 epochs
- Batch size: 16
- Effective batch size: 32
- Span masking
- Mask ratio: 0.25
- 73k sequences
- Compared performance vs non-conditioned diffusion.

1. with T=8 ; 

| Mask Type | Ratio | Train Accuracy% | Validation Accuracy% | Train Loss | Validation Loss |
|------------|--------|----------------|---------------------|------------|-----------------|
| Random     | 0.10   |           |                |      |           |
| Random     | 0.25   |           |                |      |           |
| Random     | 0.40   |           |                |      |           |
| Span       | 0.10   |           |                |      |           |
| Span       | 0.25   | 42.65          | 42.89               | 3.3225     | 3.4671          |
| Span       | 0.40   |           |                |      |           |


2. with T=12; best score with dropout 0

| Mask Type | Ratio | Train Accuracy% | Validation Accuracy% | Train Loss | Validation Loss |
|------------|--------|----------------|---------------------|------------|-----------------|
| Random     | 0.10   |                |                     |            |                 |
| Random     | 0.25   |     71.24      | 69.50               | 1.3612     | 1.5401          |
| Random     | 0.40   |                |                     |            |                 |
| Span       | 0.10   |  49.84         |  50.82              |  2.9007    |  2.9259         |
| Span       | 0.25   | 49.57          | 48.97               | 2.8369     | 3.0832          |
| Span       | 0.40   |  46.55         | 46.49               |  3.0933    |   3.2812        |


---
#### Continuing with span masking 10%

### 9️⃣ Full Diffusion Experiments
- Mask types: Span & Random
- Mask ratios: 10%, 25%, 40%
- Conditioning dropout: 0.0, 0.1
- Batch size: 16 (effective 32)
- Gradient accumulation enabled
- Evaluated robustness across corruption levels.

#### Span masking 10% with conditional dropout results - 
Train Loss: 2.9033
Train Accuracy: 0.4980
Validation Loss: 2.9092
Validation Accuracy: 0.5097


### 🔟 Inference Experiments
Tested decoding strategies:

- Temperature values: 0.8, 1.0, 1.2
- Top-k sampling: 0, 20, 50

Goal:
- Balance between diversity and accuracy.
- Improve text coherence in inpainting.

---

### 1️⃣1️⃣ User Interface
- Built Gradio-based UI:
  - Paste text
  - Auto-mask spans
  - Diffusion-based fill
  - Highlight reconstructed tokens
- Demonstrates qualitative results interactively.

---

## 📊 Final Objective

Target: **35%+ masked-token accuracy**  
Approach:
- Scaling dataset
- Diffusion modeling
- Mask conditioning
- Gradient accumulation
- Controlled decoding
