import torch


class DiscreteDiffusionForward:
    """
    Forward diffusion process for discrete tokens.
    Replaces tokens with [MASK] gradually over T steps.
    """

    def __init__(self, T, mask_token_id, beta_start=0.05, beta_end=0.40):
        self.T = T
        self.mask_token_id = mask_token_id

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T)

        # Cumulative corruption probability
        self.alpha_bars = torch.cumsum(self.betas, dim=0)
        self.alpha_bars = torch.clamp(self.alpha_bars, max=0.95)

    def to(self, device):
        self.alpha_bars = self.alpha_bars.to(device)
        return self
    
    def sample_timestep(self, batch_size, device):
        """
        Sample random timestep t ∈ [1, T]
        """
        return torch.randint(3, 7, (batch_size,), device=device)

    def corrupt(self, x0, t):
        """
        Vectorized corruption.
        x0: (batch_size, seq_len)
        t:  (batch_size,)
        """

        batch_size, seq_len = x0.shape
        device = x0.device

        # Get alpha_t for each sample
        alpha_bars = self.alpha_bars
        alpha_t = alpha_bars[t - 1]

        # Expand to match sequence
        alpha_t = alpha_t.unsqueeze(1).expand(-1, seq_len)

        # Generate random mask
        random_tensor = torch.rand(batch_size, seq_len, device=device)

        mask = random_tensor < alpha_t

        x_t = x0.clone()
        x_t[mask] = self.mask_token_id

        return x_t