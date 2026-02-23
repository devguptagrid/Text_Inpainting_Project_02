from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.masking import apply_masking


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    # Load and clean dataset
    dataset = load_wikitext()
    train_dataset = clean_dataset(dataset["train"])

    # Tokenizer
    tokenizer = get_tokenizer()

    # Tokenize
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)

    # Create fixed sequences
    train_sequences = create_fixed_length_sequences(
        tokenized_train,
        seq_len=256
    )

    # Take one example
    sample_sequence = train_sequences[0]

    # Test masking variants
    for mask_type in ["random", "span"]:
        for ratio in [0.10, 0.25, 0.40]:

            masked_input, target, mask_positions = apply_masking(
                sample_sequence,
                mask_token_id=tokenizer.mask_token_id,
                mask_type=mask_type,
                mask_ratio=ratio,
            )

            print(f"\nMask type: {mask_type}, Ratio: {ratio}")
            print("Total masked tokens:", mask_positions.sum().item())