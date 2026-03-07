# pix2pix-uint16 (PyTorch)

## About

This repository is a fork of [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) adapted for high-bit-depth image data. The original license can be found in [LICENSE](./LICENSE). This repository does not include support for CycleGAN or other models.

Many scientific imaging modalities store images as 16-bit integer data. The popular PyTorch pix2pix implementation assumes 8-bit RGB images, which can result in loss of dynamic range when loading images for training and evaluation.

This fork modifies the data pipeline so that high-bit-depth images can be used for training and inference without unnecessary precision loss.

## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

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
conda activate pix2pix-uint
```

### pix2pix train/test


- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with training script
- Train a model:

```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/<your-dataset> --name <your-dataset> --model pix2pix --direction BtoA  --use_wandb
```

- Test the model (`bash ./scripts/test_pix2pix.sh`):

```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/<your-dataset> --name <your-dataset> --model pix2pix --direction BtoA
```

### Multi-GPU training

To train a model on multiple GPUs, please use `torchrun --nproc_per_node=4 train.py ...` instead of `python train.py ...`. We also need to use synchronized batchnorm by setting `--norm sync_batch` (or `--norm sync_instance` for instance normgalization). The `--norm batch` is not compatible with DDP.


## Citation

Citation for the original pix2pix work.

```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
