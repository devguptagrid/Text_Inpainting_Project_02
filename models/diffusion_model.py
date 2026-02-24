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
        """
        input_ids: (batch_size, seq_len)
        t:         (batch_size,)
        """

        # Get base embeddings from BERT
        embeddings = self.bert_mlm.bert.embeddings(input_ids)

        # Get timestep embeddings
        t_embed = self.timestep_embedding(t)  # (batch_size, hidden_dim)

        # Expand timestep embedding across sequence length
        t_embed = t_embed.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        embeddings = embeddings + t_embed

        # Forward through encoder
        encoder_outputs = self.bert_mlm.bert.encoder(
            embeddings,
            attention_mask=attention_mask
        )

        sequence_output = encoder_outputs.last_hidden_state

        # MLM head
        logits = self.bert_mlm.cls(sequence_output)

        return logits