import torch
import torch.nn as nn
from transformers import BertForMaskedLM


class DiffusionBert(nn.Module):
    """
    BERT-based denoiser conditioned on diffusion timestep.
    """

    def __init__(self, T):
        super().__init__()

        self.T = T

        # Load pretrained MLM model
        self.bert_mlm = BertForMaskedLM.from_pretrained("bert-base-uncased")

        hidden_dim = self.bert_mlm.config.hidden_size

        # Timestep embedding
        self.timestep_embedding = nn.Embedding(T, hidden_dim)

    def forward(self, input_ids, t, attention_mask=None):

        # Get token embeddings
        token_embeddings = self.bert_mlm.bert.embeddings(input_ids)

        # Get timestep embeddings
        t_embed = self.timestep_embedding(t)  # (batch_size, hidden_dim)
        t_embed = t_embed.unsqueeze(1)        # (batch_size, 1, hidden_dim)

        # Add timestep to token embeddings
        embeddings = token_embeddings + t_embed

        # Call full BERT model properly
        outputs = self.bert_mlm.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        # MLM head
        logits = self.bert_mlm.cls(sequence_output)

        return logits