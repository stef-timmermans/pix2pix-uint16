# pix2pix-uint16 (PyTorch)

A `pix2pix` fork enabling training and inference on high-bit-depth scientific images (8-, 16-, and 32-bit).
## About

This repository is a fork of [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) adapted for high-bit-depth image data. The original license can be found in [LICENSE](./LICENSE). This repository does not include support for CycleGAN or other models.

Many scientific imaging modalities store images as 16-bit integer data. The popular PyTorch `pix2pix` implementation assumes 8-bit RGB images, which can result in loss of dynamic range when loading images for training and evaluation.

This fork modifies the data pipeline so that high-bit-depth images can be used for training and inference without unnecessary precision loss.

## Features

- Support for `uint8` / `uint16` / `uint32` training and inference  
- Avoids implicit 8-bit normalization commonly performed in computer vision pipelines
- Single-channel and multi-channel image translation  
- TIFF output support for scientific workflows  
- Optional foreground-aware reconstruction loss  
- Distributed Data Parallel (DDP) training  

## Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA + cuDNN (recommended) or CPU

## Installation

- Clone this repo:

```bash
git clone https://github.com/stef-timmermans/pix2pix-uint16.git
cd pix2pix-uint16
```

- Install [PyTorch](http://pytorch.org) and other dependencies. For Conda users, you can create a new Conda environment by

```bash
conda env create -f environment.yml
```

and then activate the environment by

```bash
conda activate pix2pix-uint16
```

## Required Arguments

```
--dtype {uint8,uint16,uint32} \     # input data type
--input_nc {1,3} \                  # input channel count
--output_nc {1,3}                   # output channel count
```

## Dataset Structure

`--dataroot` must point to a dataset directory structured as follows:

```
dataroot/
├── A/
│   ├── train/
│   ├── val/
│   └── test/
├── B/
│   ├── train/
│   ├── val/
│   └── test/
└── AB/
    ├── train/
    ├── val/
    └── test/
```

`AB` is the only folder actually used during training/evaluation, so if it has been properly prepared, the other directories can be cleared.

## Training

```
python train.py \
    --dataroot ./datasets/<dataset> \
    --name <experiment> \
    --model pix2pix \
    --direction AtoB \
    --dtype uint16 \
    --input_nc 1 \
    --output_nc 1 \
    --use_wandb
```

## Testing

```
python test.py \
    --dataroot ./datasets/<dataset> \
    --name <experiment> \
    --model pix2pix \
    --direction AtoB \
    --dtype uint16 \
    --input_nc 1 \
    --output_nc 1
```

## Foreground-Aware Reconstruction Loss

Scientific images often contain large low-signal background regions. Standard L1 reconstruction loss treats all pixels equally, which can cause the model to prioritize background accuracy over biologically relevant structures.

This repository includes an optional foreground-aware reconstruction loss that increases the importance of high-intensity regions during training.

This can improve learning when:

- The signal of interest occupies a small fraction of the image  
- Background dominates the loss  
- Precise localization of structures is required  

To enable the foreground-aware reconstruction loss:

```
--recon_loss foreground_aware
```

To tune the loss weighting behaviour, the following hyperparameters can be used (see [losses.py](./models/losses.py) for implementation details):

```
--background_percentile 5
--min_importance 0.2
--max_importance 3
--importance_scale 1000
--importance_gamma 2
```

## Multi‑GPU

This repository supports PyTorch Distributed Data Parallel (DDP) training.

To launch multi-GPU training, use **torchrun** instead of `python`.

Example (4 GPUs):

```
torchrun --nproc_per_node=4 train.py \
    --your_args
```

## Citation

Citation for the original `pix2pix` work.

```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
