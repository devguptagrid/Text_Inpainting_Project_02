import torch
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    """
    Dataset for diffusion training.
    Returns clean sequences (x0).
    No masking here — diffusion forward process handles corruption.
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.sequences[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": input_ids.clone()  # x0 target
        }