import torch


def select_device() -> torch.device:
    """Return the best available torch.device in priority: MPS, CUDA, CPU."""
    # Prefer Apple Metal if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Then prefer CUDA if available
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Fallback to CPU
    return torch.device("cpu")


device: torch.device = select_device()
