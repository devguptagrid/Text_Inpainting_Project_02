# Diffusion-Based Text Inpainting using Discrete Diffusion (D3PM)

## Overview

This project implements a **Discrete Diffusion Probabilistic Model (D3PM-style)** for span-based text inpainting using a pretrained BERT backbone.

The development progressed through:

1. Transformer from scratch
2. Pretrained BERT baseline
3. Diffusion without mask conditioning
4. Diffusion with explicit mask conditioning
5. Conditioning dropout
6. Decoding experiments (temperature & top-k)
7. Evaluation (Accuracy, BLEU, ROUGE-L)
8. Interactive Gradio UI

---

# Dataset

We use **WikiText-2 (raw version)** from HuggingFace:

```python
load_dataset("wikitext", "wikitext-2-raw-v1")
```

### Original Dataset Sizes

| Split       | Samples |
|------------|----------|
| Train      | 36,718   |
| Validation | 3,760    |
| Test       | 4,358    |

### After Cleaning
(Removed empty lines, headers, short fragments <10 characters)

| Split       | Samples |
|------------|----------|
| Train      | 23,547   |
| Validation | 2,454    |
| Test       | 2,850    |

---

# Preprocessing Pipeline

1. Tokenization: `bert-base-uncased`
2. Fixed-length chunking (256 tokens)
3. Sliding window expansion → ~73k sequences
4. Masking strategies:
   - Random token masking
   - Contiguous span masking
5. Dynamic masking during training
6. Fixed masking during validation/test

---

# Phase 1 — Transformer From Scratch

Custom Transformer encoder built using PyTorch:

- Hidden size: 256 → 384
- Layers: 4 → 6
- Attention heads: 4 → 6
- Masked cross-entropy loss

### Result
Masked-token accuracy: **~5–6%**

### Conclusion
WikiText-2 (~2M tokens) is insufficient to train a strong language model from scratch.

---

# Phase 2 — Pretrained BERT Baseline

Used:

```python
BertForMaskedLM.from_pretrained("bert-base-uncased")
```

Fine-tuned on masked token prediction.

### Results (9k sequences, 3 epochs, batch=32)

 Mask Type | Ratio | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|------------|--------|----------------|---------------------|------------|-----------------|
| Random     | 0.10   | 57.83          | 57.36               | 2.1694     | 2.2301          |
| Random     | 0.25   | 53.07          | 51.85               | 2.4602     | 2.6114          |
| Random     | 0.40   | 45.01          | 43.90               | 3.0190     | 3.2059          |
| Span       | 0.10   | 20.45          | 21.23               | 5.1082     | 5.1766          |
| Span       | 0.25   | 19.44          | 19.94               | 5.0969     | 5.2521          |
| Span       | 0.40   | 17.51          | 17.91               | 5.2213     | 5.4480          |

**Observation:** Random masking is significantly easier than span masking.

### Results (73k sequences, 3 epochs, batch=32, span masking, span ratio - 0.25)
Train Loss: **4.5502**

Train Accuracy: **21.79%**

Validation Loss: **5.4015**

Validation Accuracy: **19.89%**

---

# Phase 3 — Diffusion Model (Without Mask Conditioning)

Implemented discrete forward corruption:

- T diffusion timesteps
- Gradual masking
- Reverse denoising using BERT


### Setup 1
- Training:
  - 6 epochs
  - Batch size: 16
  - Span masking
  - Mask ratio: 0.25
  - Mask type: Span
  - 9k sequences
- Goal: Compare diffusion vs single-step MLM baseline.

Train Loss: **5.0208**

Train Accuracy: **22.80%**

Validation Loss: **4.9633**

Validation Accuracy: **24.39%**

### Setup 2
- T = 12
- Span masking = 0.25
- Batch size = 16
- Gradient accumulation = 2 (effective batch size = 32)
- 73k sequences
- 6 epochs

### Result

Train Loss: **4.5649**

Train Accuracy: **26.08%**

Validation Loss: **4.9749**

Validation Accuracy: **24.58%**

---

# Phase 4 — Diffusion With Mask Conditioning

Added explicit mask embeddings so the model knows which tokens were originally masked.

---

## T = 8 (Span 0.25)

Train Loss: **3.3225**

Train Accuracy: **42.65%**

Validation Loss: **3.4671**

Validation Accuracy: **42.89%**

---


## T = 12 (Best Configuration)

| Mask Type | Ratio | Train Accuracy% | Validation Accuracy% | Train Loss | Validation Loss |
|------------|--------|----------------|---------------------|------------|-----------------|
| Random     | 0.25   |     71.24      | 69.50               | 1.3612     | 1.5401          |
| Span       | 0.10   |  49.84         |  50.82              |  2.9007    |  2.9259         |
| Span       | 0.25   | 49.57          | 48.97               | 2.8369     | 3.0832          |
| Span       | 0.40   |  46.55         | 46.49               |  3.0933    |   3.2812        |

---

# Conditioning Dropout

Added conditioning dropout (0.1) to reduce over-reliance on mask embeddings.

Best Span 10% Result:

Train Loss: **2.9033**

Train Accuracy: **49.80%**

Validation Loss: **2.9092**

Validation Accuracy: **50.97%**


---


# Final Selected Model

Configuration:

- Mask type: Span
- Mask ratio: 0.10
- Timestep: 12
- Conditioning dropout: 0.1
- Temperature: 0.8
- Top-k: 20

---

# Test Results

Test Loss: **2.9221**  
Test Accuracy: **51.14%**

---

## Masked-Only Evaluation

| Metric | Score |
|---------|--------|
| Masked BLEU | 0.0826 |
| Masked ROUGE-L | 0.3484 |

BLEU remains low due to strict n-gram matching.  
ROUGE-L better captures structural similarity in span reconstruction.

---

# Inference Decoding Experiments

Tested:

- Temperature: 0.8, 1.0, 1.2
- Top-k: 0, 20, 50

Best decoding configuration:

```
Temperature = 0.8
Top-k = 20
```

Balanced diversity and coherence.

---


# Gradio UI

Interactive interface allows:

- Paste input text
- Auto span masking
- Diffusion-based reconstruction
- Highlight reconstructed tokens
- Adjustable temperature
- Adjustable top-k

Video_link :  https://drive.google.com/file/d/1Dli7Pqbi3G5PoX9ktUoxGgUH2dXZUfCr/view


# Project Structure

```
TEXT-INPAINTING_02/
│
├── data/
│   ├── dataset.py
│   ├── diffusion_dataset.py
│   ├── load_data.py
│   ├── masking.py
│   └── preprocessing.py
│
├── diffusion/
│   └── forward_process.py
│
├── evaluation/
│   ├── bleu.py
│   ├── metrics.py
│   └── rouge.py
│
├── inference/
│   ├── inpaint.py
│   └── reverse_diffusion.py
│
├── models/
│   ├── diffusion_model.py
│   └── transformer.py
│
├── notebooks/
│   └── 01_data_analysis.ipynb
│
├── training/
│   ├── diffusion_trainer.py
│   ├── loss.py
│   └── trainer.py
│
├── utils/
│   ├── device.py
│   └── seed.py
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Folder Responsibilities

### `data/`
Handles dataset loading, preprocessing, tokenization, and masking logic.

### `diffusion/`
Implements the forward corruption process for discrete diffusion.

### `models/`
Contains:
- Baseline Transformer
- DiffusionBert model with timestep + mask conditioning

### `training/`
Training loops for:
- Baseline model
- Diffusion model
- Gradient accumulation logic

### `inference/`
Reverse diffusion sampling and inpainting pipeline.

### `evaluation/`
Implements:
- BLEU score
- ROUGE-L
- Accuracy metrics

### `utils/`
Utility functions:
- Device management (CPU / MPS)
- Seed setting for reproducibility

### `app.py`
Gradio-based UI for interactive text inpainting.

### `main.py`
Entry point for:
- Training
- Validation
- Testing
- BLEU / ROUGE evaluation
- Inference mode

---


# Requirements

```
torch
transformers
datasets
tokenizers
numpy
pandas
tqdm
gradio
nltk
rouge-score
```

---



# Final Outcome

Target: **35%+ masked-token accuracy**

Final Achieved: **50%+ masked-token accuracy**

This project demonstrates a complete diffusion-based text inpainting pipeline with training, evaluation, and interactive deployment.


# How to Run the Project

Follow the steps below to reproduce the experiments and run the UI locally.

---

## 1. Clone the Repository

```bash
git clone https://github.com/devguptagrid/Text_Inpainting_Project_02.git
cd Text_Inpainting_Project_02
```
## 2. Create Virtual Environment
#### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Running the project 
### Train Baseline Model
```bash
python main.py --mode train_baseline
```

### Train Diffusion Model
```bash
python main.py --mode diffusion
```

### Run evaluation
```bash
python main.py --mode test
```

### Run inference
```bash
python main.py --mode inference
```

### Run Gradio UI
```bash
python app.py
```

### Using the UI

#### Steps:

- Paste any input sentence

- The system automatically applies span masking

- The diffusion model reconstructs the masked tokens

- Reconstructed tokens appear highlighted in green

- You can also adjust:
    - Temperature – controls randomness

    - Top-k – restricts tokens sampling to top-k candidates