## Defines the diffusion-based BERT model with timestep and mask conditioning for denoising.

import torch
import torch.nn as nn
from transformers import BertForMaskedLM


class DiffusionBert(nn.Module):
    """
    BERT-based denoiser conditioned on diffusion timestep.
    """

    def __init__(self, T, conditioning_dropout=0.0):
        super().__init__()

        self.T = T
        self.conditioning_dropout = conditioning_dropout

        # Load pretrained Bert MLM model
        self.bert_mlm = BertForMaskedLM.from_pretrained("bert-base-uncased")

        self.hidden_dim = self.bert_mlm.config.hidden_size ##internal vector dimension (768 for bert-base) used throughout BERT for token representations, attention, feedforward layers, and output projection

        # Timestep embedding
        self.timestep_embedding = torch.nn.Embedding(T, self.hidden_dim)
        # Mask embedding (0 = clean, 1 = corrupted)
        self.mask_embedding = torch.nn.Embedding(2, self.hidden_dim)
    def forward(self, x_t, t_embed, mask_positions, attention_mask=None):

        # 1️⃣ Get full BERT embeddings (this keeps position + layernorm + dropout)
        bert_embeddings = self.bert_mlm.bert.embeddings(input_ids=x_t)

        # 2️⃣ Timestep embedding
        timestep_embeds = self.timestep_embedding(t_embed)  # (batch, hidden)
        timestep_embeds = timestep_embeds.unsqueeze(1)      # (batch, 1, hidden)

        # 3️⃣ Mask embedding (0 = clean, 1 = corrupted)
        mask_embeds = self.mask_embedding(mask_positions.long())

        if self.training and self.conditioning_dropout > 0:
            dropout_mask = ( ## Randomly drop conditioning information during training to improve robustness.
                torch.rand(mask_embeds.shape[0], device=mask_embeds.device) 
                < self.conditioning_dropout 
            ).float().unsqueeze(1).unsqueeze(2)

            mask_embeds = mask_embeds * (1 - dropout_mask) ## During training, randomly zero out the mask embeddings for some samples based on the conditioning_dropout probability.

        # 4️⃣ Combine all embeddings
        embeddings = bert_embeddings + timestep_embeds + mask_embeds

        # 5️⃣ Forward through BERT encoder correctly
        outputs = self.bert_mlm.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.bert_mlm.cls(sequence_output)

        return logits