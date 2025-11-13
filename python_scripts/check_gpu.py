import torch

if torch.cuda.is_available():
    print(f"--- SUCCESS ---")
    print(f"Torch can see your GPU!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print(f"--- FAILED ---")
    print(f"Torch CANNOT find a compatible GPU (NVIDIA CUDA).")
    print(f"The MedGemma model will fall back to your CPU, which is extremely slow.")