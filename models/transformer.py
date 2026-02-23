import torch
import torch.nn as nn


class TransformerDenoiser(nn.Module): ##defining a pytorch neural network
   
   ## Simple Transformer Encoder-based denoiser for masked token reconstruction similar to BERT

    def __init__(
        self, ## constructor method to initialize the model's parameters and layers
        vocab_size=30522,  ## standard BERT vocab size, number of possible tokens the model can predict.
        hidden_dim=256, ## smaller hidden dimension for efficiency
        num_layers=4, ##number of stacked transformer encoder layers
        num_heads=4, ## number of attention heads in each transformer layer
        max_seq_len=256,
        dropout=0.1, ##regularization to prevent overfitting
    ):
        super().__init__() ##initializes the parent class (nn.Module) to set up the neural network structure

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, ## the number of expected features in the input (hidden dimension)
            nhead=num_heads, ## the number of heads in the multiheadattention models
            dim_feedforward=hidden_dim * 4, ## the dimension of the feedforward network model (usually 4 times the hidden dimension)
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder( ## stacks multiple transformer encoder layers to form the full transformer model
            encoder_layer,
            num_layers=num_layers,
        )

        # Final output projection to vocabulary
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        input_ids: (B, seq_len)
        returns: logits (B, seq_len, vocab_size)
        """

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Position indices
        positions = torch.arange(
            0, seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        pos_emb = self.position_embedding(positions)

        # Combine token + position embeddings
        x = token_emb + pos_emb

        # Transformer encoding
        x = self.transformer(x)

        # Project to vocabulary
        logits = self.output_layer(x)

        return logits ##raw scores for each possible token before converting to probabilities.