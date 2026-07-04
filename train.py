"""General-purpose training script for image-to-image translation.

This script works for pix2pix with an aligned dataset (using '--dataset_mode aligned')
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pix2pix model:
        python train.py --dataroot ./datasets/<your-dataset> --name <your-dataset> --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
"""

import time
from pathlib import Path
import torch.distributed as dist
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp
from util.wandb_helper import finish_run


def prune_old_epoch_checkpoints(save_dir: Path, current_epoch: int, keep_last_n: int) -> None:
    """Delete numbered epoch checkpoints older than the most recent N epochs."""
    if keep_last_n <= 0:
        return

    min_epoch_to_keep = current_epoch - keep_last_n + 1
    for checkpoint_path in save_dir.glob("*_net_*.pth"):
        stem = checkpoint_path.stem
        epoch_token, _, _ = stem.partition("_net_")
        if epoch_token == "latest":
            continue
        try:
            checkpoint_epoch = int(epoch_token)
        except ValueError:
            continue
        if checkpoint_epoch < min_epoch_to_keep:
            checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    if hasattr(visualizer, "wandb_run") and visualizer.wandb_run is not None:
        visualizer.wandb_run.summary["train/num_images"] = dataset_size
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if visualizer.use_wandb:
                    visualizer.plot_current_losses(total_iters, losses)
                if getattr(visualizer, "wandb_run", None) is not None:
                    visualizer.wandb_run.log({"train/epoch": epoch}, step=total_iters)

            if not opt.save_latest_only and total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch
        if getattr(visualizer, "wandb_run", None) is not None:
            generator_lr = model.optimizer_G.param_groups[0]["lr"]
            discriminator_lr = model.optimizer_D.param_groups[0]["lr"]
            lr_metrics = {
                "train/lr_G": generator_lr,
                "train/lr_D": discriminator_lr,
                "train/epoch_complete": epoch,
            }
            visualizer.wandb_run.log(lr_metrics, step=total_iters)

        if opt.save_latest_only:
            if epoch == opt.n_epochs + opt.n_epochs_decay:
                print(f"saving the final latest model at the end of epoch {epoch}, iters {total_iters}")
                model.save_networks("latest")
        else:
            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
                model.save_networks("latest")
                model.save_networks(epoch)
                prune_old_epoch_checkpoints(model.save_dir, epoch, opt.save_last_n_epochs)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} 	 Time Taken: {time.time() - epoch_start_time:.0f} sec")

    if not dist.is_initialized() or dist.get_rank() == 0:
        finish_run(getattr(visualizer, "wandb_run", None))

    cleanup_ddp()
