"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation helpers such as get_params and get_transform.
"""

import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import functools

from util.types import dtype_max


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def _interp_mode(interp_method):
    if interp_method == transforms.InterpolationMode.NEAREST:
        return "nearest"
    if interp_method == transforms.InterpolationMode.BILINEAR:
        return "bilinear"
    if interp_method == transforms.InterpolationMode.BICUBIC:
        return "bicubic"
    # torch interpolate does not support lanczos
    return "bicubic"


def _resize_tensor(img, size_hw, interp_method):
    mode = _interp_mode(interp_method)
    img = img.unsqueeze(0)  # 1 x C x H x W
    if mode in ("bilinear", "bicubic"):
        img = F.interpolate(img, size=size_hw, mode=mode, align_corners=False)
    else:
        img = F.interpolate(img, size=size_hw, mode=mode)
    return img.squeeze(0)


def _scale_width_tensor(img, target_size, crop_size, interp_method):
    _, h, w = img.shape
    if w == target_size and h >= crop_size:
        return img
    new_w = target_size
    new_h = int(max(target_size * h / w, crop_size))
    return _resize_tensor(img, (new_h, new_w), interp_method)


def _make_power_2_tensor(img, base=4, interp_method=transforms.InterpolationMode.BICUBIC):
    _, h, w = img.shape
    new_h = int(round(h / base) * base)
    new_w = int(round(w / base) * base)
    if new_h == h and new_w == w:
        return img
    return _resize_tensor(img, (new_h, new_w), interp_method)


def _crop_tensor(img, pos, size):
    x1, y1 = pos
    return img[:, y1:y1 + size, x1:x1 + size]


def _flip_tensor(img):
    return torch.flip(img, dims=[2])  # flip width dimension


def _to_tensor_from_array(img, grayscale=False, opt=None):
    img = np.asarray(img)

    if img.ndim == 2:
        img = img[..., None]  # H x W -> H x W x 1
    elif img.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {img.shape}")

    if grayscale and img.shape[2] != 1:
        img = img[..., :1]

    if np.issubdtype(img.dtype, np.integer):
        if opt is None:
            raise ValueError("opt must be provided for integer image normalization")
        max_val = dtype_max(opt.dtype)
        img = img.astype(np.float32) / float(max_val)
    else:
        img = img.astype(np.float32)

    return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()


def _prepare_input_tensor(img, grayscale=False, opt=None):
    if isinstance(img, torch.Tensor):
        tensor = img.float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
    else:
        tensor = _to_tensor_from_array(img, grayscale=grayscale, opt=opt)

    if grayscale and tensor.shape[0] != 1:
        tensor = tensor[:1, :, :]

    return tensor


def _apply_transform(img, opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    tensor = _prepare_input_tensor(img, grayscale=grayscale, opt=opt)

    if "resize" in opt.preprocess:
        tensor = _resize_tensor(tensor, (opt.load_size, opt.load_size), method)
    elif "scale_width" in opt.preprocess:
        tensor = _scale_width_tensor(tensor, opt.load_size, opt.crop_size, method)

    if "crop" in opt.preprocess:
        if params is None:
            _, h, w = tensor.shape
            x = random.randint(0, max(0, w - opt.crop_size))
            y = random.randint(0, max(0, h - opt.crop_size))
            tensor = _crop_tensor(tensor, (x, y), opt.crop_size)
        else:
            tensor = _crop_tensor(tensor, params["crop_pos"], opt.crop_size)

    if opt.preprocess == "none":
        tensor = _make_power_2_tensor(tensor, base=4, interp_method=method)

    if not opt.no_flip:
        if params is None:
            if random.random() > 0.5:
                tensor = _flip_tensor(tensor)
        elif params["flip"]:
            tensor = _flip_tensor(tensor)

    if convert:
        tensor = tensor * 2.0 - 1.0  # map [0, 1] -> [-1, 1]

    return tensor


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    return functools.partial(
        _apply_transform,
        opt=opt,
        params=params,
        grayscale=grayscale,
        method=method,
        convert=convert,
    )
