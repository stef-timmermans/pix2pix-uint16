import numpy as np
from types import SimpleNamespace
from base_dataset import get_transform

# Fake training options
opt = SimpleNamespace(
    preprocess="none",
    load_size=256,
    crop_size=256,
    no_flip=True,
)

# Create synthetic uint16 image
img = np.linspace(0, 65535, 64 * 64, dtype=np.uint16).reshape(64, 64)

transform = get_transform(
    opt,
    params={"crop_pos": (0, 0), "flip": False},
    grayscale=True,
)

tensor = transform(img)

print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)
print("Tensor min:", tensor.min().item())
print("Tensor max:", tensor.max().item())
