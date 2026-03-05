## Implements the baseline transformer model used for comparison with the diffusion model.

import torch
import torch.nn as nn
from transformers import BertForMaskedLM

class BertDenoiser(nn.Module):

    def __init__(self): ## Initializes the BertDenoiser model by loading a pretrained BERT model for masked language modeling (MLM) from the Hugging Face Transformers library.
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask=None): ## Defines the forward pass of the Bert
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits