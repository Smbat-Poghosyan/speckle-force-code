import torch

print("PyTorch:", torch.__version__)
print("XPU available:", torch.xpu.is_available() if hasattr(torch, "xpu") else False)

if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("GPU:", torch.xpu.get_device_name(0))