import os
import numpy as np
import tifffile
from types import SimpleNamespace
from torch.utils.data import DataLoader

from aligned_dataset import AlignedDataset

# Create temporary dataset
root = "tmp_dataset"
train_dir = os.path.join(root, "train")
os.makedirs(train_dir, exist_ok=True)

h, w = 32, 32

A = np.zeros((h, w), dtype=np.uint16)
B = np.full((h, w), 65535, dtype=np.uint16)

AB = np.concatenate([A, B], axis=1)

path = os.path.join(train_dir, "sample.tiff")
tifffile.imwrite(path, AB)

# Fake options
opt = SimpleNamespace(
    dataroot=root,
    phase="train",
    max_dataset_size=float("inf"),
    load_size=32,
    crop_size=32,
    input_nc=1,
    output_nc=1,
    direction="AtoB",
    preprocess="none",
    no_flip=True,
)

dataset = AlignedDataset(opt)

sample = dataset[0]

print("A shape:", sample["A"].shape)
print("B shape:", sample["B"].shape)

print("A min/max:", sample["A"].min().item(), sample["A"].max().item())
print("B min/max:", sample["B"].min().item(), sample["B"].max().item())

# Debug the loader
loader = DataLoader(dataset, batch_size=1)
batch = next(iter(loader))

print("Batch A:", batch["A"].shape)
print("Batch B:", batch["B"].shape)