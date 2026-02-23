from utils.seed import set_seed
from utils.device import get_device


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Device:", device)