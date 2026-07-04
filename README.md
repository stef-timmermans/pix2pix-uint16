# pix2pix-uint16 (PyTorch)

A `pix2pix` fork enabling training and inference on high-bit-depth scientific images (8-, 16-, and 32-bit).
## About

This repository is a fork of [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) adapted for high-bit-depth image data. The original license can be found in [LICENSE](./LICENSE). This repository is intentionally scoped to paired-image `pix2pix` workflows.

Many scientific imaging modalities store images as 16-bit integer data. The popular PyTorch `pix2pix` implementation assumes 8-bit RGB images, which can result in loss of dynamic range when loading images for training and evaluation.

This fork modifies the data pipeline so that high-bit-depth images can be used for training and inference without unnecessary precision loss.

## Features

- Support for `uint8` / `uint16` / `uint32` training and inference  
- Avoids implicit 8-bit normalization commonly performed in computer vision pipelines
- Single-channel and multi-channel image translation  
- TIFF output support for scientific workflows  
- Optional foreground-aware reconstruction loss  
- Distributed Data Parallel (DDP) training

## Scope

- Supported model: `pix2pix`
- Supported dataset mode: `aligned`
- Supported image dtypes: `uint8`, `uint16`, `uint32`
- Supported input/output channel counts: `1` or `3`

CycleGAN-specific training paths are not part of this fork.

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

Optional source-domain normalization can be enabled with:

```bash
--normalize_source \
--source-norm-low 1 \
--source-norm-high 99
```

Optional Weights & Biases logging can be enabled with `--use_wandb`. Project, entity, mode, and image-logging behavior are available as flags if you want to override the defaults:

```bash
--wandb_project_name pix2pix-uint16 \
--wandb_entity <username-or-team> \
--wandb_mode online
```

Image logging to W&B is optional and disabled unless you pass:

```bash
--wandb_log_images
```

## Dataset Structure

`--dataroot` should usually point to the dataset root containing an `AB/` child. Passing the `AB/` directory itself is also supported for compatibility.

Preferred layout:

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

Each split contains side-by-side paired images where the left half is domain `A` and the right half is domain `B`.

The `A/` and `B/` source folders are only needed while preparing paired data. Training and evaluation read from the paired split directories.

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

Useful training flags:

```bash
--lr 0.0002 \
--lr_G 0.0002 \
--lr_D 0.0002 \
--lambda_GAN 1.0 \
--lambda_L1 100.0 \
--save_latest_only
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

Useful evaluation flags:

```bash
--results_dir ./results \
--phase val \
--epoch latest \
--no_html \
--tiled_inference \
--tile_size 256 \
--tile_stride 256 \
--compute_eval_loss \
--skip_save_images
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

To tune the loss weighting behavior, use the following hyperparameters (see [losses.py](./models/losses.py) for implementation details):

```
--background_percentile 5
--foreground_margin 500
--fg_weight 20
--bg_weight 0.5
```

## Multi‑GPU

This repository supports PyTorch Distributed Data Parallel (DDP) training.

To launch multi-GPU training, use **torchrun** instead of `python`.

Example (4 GPUs):

```
torchrun --nproc_per_node=4 train.py \
    --your_args
```

For multi-GPU runs, prefer `--norm syncbatch` or `--norm instance`.

## W&B Authentication

This repo reads standard W&B environment variables if present, including:

- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `WANDB_MODE`
- `WANDB_BASE_URL`

You can also authenticate separately with `wandb login`. This repository does not require any repo-specific secret file.

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
