##Utility functions to configure and detect available hardware (CPU or MPS).
import torch


def get_device():
    """
    Select best device for Mac:
    - Apple MPS if available
    - CPU otherwise
    """

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple MPS acceleration")

    else:
        device = torch.device("cpu")
        print("[INFO] MPS not available, using CPU")

    return device