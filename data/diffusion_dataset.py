##Creates the dataset specifically for diffusion training, including masked inputs and target tokens.

import torch
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    """
    Dataset for diffusion training.
    Returns clean sequences (x0).
    No masking here — diffusion forward process handles corruption.
    """

    def __init__(self, sequences): ## sequences is a list of tokenized sequences (length=256)
        self.sequences = sequences

    def __len__(self): ## returns number of training samples
        return len(self.sequences)

    def __getitem__(self, idx): ## returns masked input, target ids, and mask positions for a given index. If dynamic masking is enabled, it applies masking. Otherwise, it retrieves precomputed masked data.
        input_ids = torch.tensor(self.sequences[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": input_ids.clone()  # x0 target
        }