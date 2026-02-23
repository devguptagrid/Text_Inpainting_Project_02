from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    dataset = load_wikitext()
    train_dataset = clean_dataset(dataset["train"])

    tokenizer = get_tokenizer()

    tokenized_train = tokenize_dataset(train_dataset, tokenizer)

    train_sequences = create_fixed_length_sequences(
        tokenized_train,
        seq_len=256
    )

    print("\n[INFO] Example sequence length:", len(train_sequences[0]))
    print("[INFO] First 20 tokens:", train_sequences[0][:20])