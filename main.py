from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    dataset = load_wikitext()

    train_dataset = clean_dataset(dataset["train"])
    val_dataset = clean_dataset(dataset["validation"])
    test_dataset = clean_dataset(dataset["test"])

    print("\n[INFO] Dataset sizes after cleaning:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    print("\n[INFO] Sample text:")
    print(train_dataset[0]["text"][:200])