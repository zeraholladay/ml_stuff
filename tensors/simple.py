# PyTorch tensors: ranks, shapes, devices, operations, autograd
import os
import sys

import torch

# Allow importing project root config when running this script directly
try:
    from config import device
except ModuleNotFoundError:  # running from subdir without package context
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import device

print(f"PyTorch: {torch.__version__}")
print(f"Using device: {device}\n")

# ---- 1) Tensors as data containers (different ranks) ----
scalar = torch.tensor(25.0)  # rank-0 (scalar)
vector = torch.tensor([3.0, 5.0, 8.0])  # rank-1 (vector)
matrix = torch.arange(12.0, dtype=torch.float32).reshape(3, 4)  # rank-2 (matrix)

# In computer vision, PyTorch commonly uses channel-first (C, H, W)
gray_image = torch.randn(28, 28)  # rank-2 (grayscale image: H, W)
color_image = torch.randn(3, 224, 224)  # rank-3 (C, H, W)
batch_of_images = torch.randn(32, 3, 224, 224)  # rank-4 (N, C, H, W)

tensors = {
    "scalar": scalar,
    "vector": vector,
    "matrix": matrix,
    "gray_image": gray_image,
    "color_image (C,H,W)": color_image,
    "batch (N,C,H,W)": batch_of_images,
}

print("RANK & SHAPE")
for name, t in tensors.items():
    print(f"{name:20s} shape={tuple(t.shape)!s:18s} rank(ndim)={t.ndim}")
print()

# ---- 2) Dtype, device, memory layout ----
print("DTYPE / DEVICE / CONTIGUITY")
print(
    f"matrix.dtype={matrix.dtype}, matrix.device={matrix.device}, contiguous={matrix.is_contiguous()}\n"
)

# Move a big tensor to GPU if available
batch_of_images = batch_of_images.to(device)
print(f"batch_of_images is on: {batch_of_images.device}\n")

# ---- 3) Mathematical operations (vectorized) ----
# Matrix multiplication
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B  # (2,4)
print("Matrix multiply shapes:", A.shape, "@", B.shape, "->", C.shape)

# Broadcasting: add a row vector (3,) to each row of a (2,3) matrix
X = torch.arange(6.0, dtype=torch.float32).reshape(2, 3)
b = torch.tensor([10.0, 20.0, 30.0])
X_plus_b = X + b
print("Broadcast add shapes:", X.shape, "+", b.shape, "->", X_plus_b.shape, "\n")

# ---- 4) Autograd (automatic differentiation) ----
# Tiny linear regression step: y = xW + b
torch.manual_seed(7)
x = torch.randn(5, 3, device=device)
true_W = torch.tensor([[2.0], [-3.0], [0.5]], device=device)
true_b = torch.tensor([0.1], device=device)
y = x @ true_W + true_b + 0.01 * torch.randn(5, 1, device=device)  # noisy targets

# Parameters with gradients
W = torch.randn(3, 1, device=device, requires_grad=True)
b_param = torch.zeros(1, device=device, requires_grad=True)

# Forward pass
pred = x @ W + b_param
loss = torch.mean((pred - y) ** 2)

print("Autograd info:")
print(f"  pred.requires_grad={pred.requires_grad}")
print(f"  pred.grad_fn={pred.grad_fn}")  # shows the node in the computation graph
print(f"  loss={loss.item():.6f}")

# Backward pass
loss.backward()
print("Gradients:")
print("  W.grad:", W.grad.flatten().tolist())
print("  b.grad:", b_param.grad.flatten().tolist())

# One simple SGD step (no optimizer object, just manual)
lr = 0.1
with torch.no_grad():
    W -= lr * W.grad
    b_param -= lr * b_param.grad
    W.grad.zero_()
    b_param.grad.zero_()

# Recompute loss after one step to show it decreases (usually)
new_loss = torch.mean((x @ W + b_param - y) ** 2).item()
print(f"loss after one SGD step: {new_loss:.6f} (was {loss.item():.6f})")
