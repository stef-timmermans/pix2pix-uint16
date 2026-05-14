"""Dataset entrypoints for the supported paired-image loaders."""

import os
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from data.aligned_dataset import AlignedDataset


DATASET_REGISTRY = {
    "aligned": AlignedDataset,
}


def get_dataset_class(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise NotImplementedError(
            f"Unsupported dataset_mode '{dataset_name}'. Supported values: {supported}."
        )
    return DATASET_REGISTRY[dataset_name]


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = get_dataset_class(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = get_dataset_class(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        # Use DistributedSampler for DDP training
        if "LOCAL_RANK" in os.environ:
            print(f'create DDP sampler on rank {int(os.environ["LOCAL_RANK"])}')
            self.sampler = DistributedSampler(self.dataset, shuffle=not opt.serial_batches)
            shuffle = False  # DistributedSampler handles shuffling
        else:
            self.sampler = None
            shuffle = not opt.serial_batches

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=shuffle, sampler=self.sampler, num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def set_epoch(self, epoch):
        """Set epoch for DistributedSampler to ensure proper shuffling"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
