import torch
from torch.utils.data import Dataset
from data.masking import apply_masking


class TextInpaintingDataset(Dataset):

    ##Applies masking dynamically per sample.


    def __init__( 
        self, 
        sequences, ## list of tokenized sequences (length=256)
        tokenizer, ## HuggingFace tokenizer
        mask_type="span", ## "span" or "random"
        mask_ratio=0.25, ## e.g. 0.1, 0.25, 0.4
        dynamic_masking=True,
    ):
     

        ## stored for later access
        self.sequences = sequences 
        self.tokenizer = tokenizer 
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.dynamic_masking = dynamic_masking

        self.mask_token_id = tokenizer.mask_token_id

        # If not dynamic → precompute masked samples
        if not self.dynamic_masking:
            self.precomputed_data = []
            for seq in self.sequences:
                masked_input, target_ids, mask_positions = apply_masking(
                    input_ids=seq,
                    mask_token_id=self.mask_token_id,
                    mask_type=self.mask_type,
                    mask_ratio=self.mask_ratio,
                )

                self.precomputed_data.append({
                    "input_ids": masked_input,
                    "target_ids": target_ids,
                    "mask_positions": mask_positions,
                })

    def __len__(self):  ##returns number of training samples
        return len(self.sequences)

    def __getitem__(self, idx):

        if self.dynamic_masking:
            input_ids = self.sequences[idx]

            masked_input, target_ids, mask_positions = apply_masking(
                input_ids=input_ids,
                mask_token_id=self.mask_token_id,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
            )

            return {
                "input_ids": masked_input, ##[101, 2023, 103, 103, 6251, ..., 102]
                "target_ids": target_ids, ##[101, 2023, 2003, 1037, 6251, ..., 102]
                "mask_positions": mask_positions, ##[False, False, True, True, False, ...]
            }
        else:
            return self.precomputed_data[idx]