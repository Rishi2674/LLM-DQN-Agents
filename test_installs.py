import torch
import gc
from transformers import AutoModel


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs available:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))


import os
from pathlib import Path

cache_dir = Path.home() / ".cache/huggingface/hub"
if cache_dir.exists():
    print("Installed models:")
    for item in os.listdir(cache_dir):
        if "models--" in item:
            print(item.replace("models--", "").replace("--", "/"))
else:
    print("No Hugging Face models found locally.")

print(torch.cuda.memory_summary(device=torch.cuda.current_device()))

# Print current GPU memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


for obj in gc.get_objects():
    if isinstance(obj, torch.nn.Module):
        print(f"Model: {type(obj).__name__}, Device: {next(obj.parameters()).device}")

total_mem = 0
for obj in gc.get_objects():
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        mem = obj.element_size() * obj.numel() / (1024**2)  # Convert bytes to MB
        total_mem += mem
        print(f"Tensor: {obj.shape}, Device: {obj.device}, Memory: {mem:.2f} MB")

print(f"Total Tensor Memory Usage: {total_mem:.2f} MB")

import sys

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key=lambda x: -x[1])[:10]:
    print(f"{name}: {size / (1024**2):.2f} MB")
