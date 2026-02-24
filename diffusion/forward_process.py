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

    def sample_timestep(self, batch_size, device):
        """
        Sample random timestep t ∈ [1, T]
        """
        return torch.randint(1, self.T + 1, (batch_size,), device=device)

    def corrupt(self, x0, t):
        """
        Generate x_t from x0.
        x0: (batch_size, seq_len)
        t:  (batch_size,)
        """

        batch_size, seq_len = x0.shape
        device = x0.device

        x_t = x0.clone()

        for i in range(batch_size):
            alpha_t = self.alpha_bars[t[i] - 1]

            mask = torch.rand(seq_len, device=device) < alpha_t
            x_t[i][mask] = self.mask_token_id

        return x_t