import torch
import torch.nn as nn
from transformers import BertForMaskedLM

class BertDenoiser(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits