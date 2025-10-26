import torch
ckpt = torch.load("checkpoints/99_model.pt", weights_only=False)
print(ckpt.keys())
