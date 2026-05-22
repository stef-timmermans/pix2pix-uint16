"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example:
    Test a pix2pix model:
        python test.py --dataroot ./datasets/<your-dataset> --name <your-dataset> --model pix2pix --direction AtoB

See options/base_options.py and options/test_options.py for more test options.
"""

import json
from pathlib import Path
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.image_logging import save_visuals_to_directory
import torch
import csv
from models.losses import reconstruction_loss
from util.wandb_helper import finish_run, init_wandb_run, log_visuals, update_config, update_summary


def tiled_test(model, data, opt):
    tile_size = opt.tile_size
    tile_stride = opt.tile_stride
    real_A = data["A"]
    _, _, h, w = real_A.shape

    assert h % tile_size == 0
    assert w % tile_size == 0
    assert tile_stride > 0
    assert tile_stride <= tile_size
    assert (h - tile_size) % tile_stride == 0
    assert (w - tile_size) % tile_stride == 0

    fake_B = None
    weights = None

    for y in range(0, h - tile_size + 1, tile_stride):
        for x in range(0, w - tile_size + 1, tile_stride):
            tile_data = dict(data)
            tile_data["A"] = real_A[:, :, y:y + tile_size, x:x + tile_size]
            if "B" in tile_data:
                tile_data["B"] = data["B"][:, :, y:y + tile_size, x:x + tile_size]

            model.set_input(tile_data)
            model.test()

            fake_tile = model.fake_B.detach()

            if fake_B is None:
                fake_B = torch.zeros(
                    fake_tile.shape[0],
                    fake_tile.shape[1],
                    h,
                    w,
                    dtype=fake_tile.dtype,
                    device=fake_tile.device,
                )
                weights = torch.zeros_like(fake_B)

            fake_B[:, :, y:y + tile_size, x:x + tile_size] += fake_tile
            weights[:, :, y:y + tile_size, x:x + tile_size] += 1

    fake_B = fake_B / weights

    visuals = {
        "real_A": real_A,
        "fake_B": fake_B,
    }
    if "B" in data:
        visuals["real_B"] = data["B"]

    return visuals, fake_B

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    wandb_run = init_wandb_run(
        opt,
        job_type="eval",
        run_name=f"{opt.name}-{opt.phase}-{opt.epoch}",
    )
    update_config(
        wandb_run,
        {
            "phase": opt.phase,
            "epoch": opt.epoch,
            "tiled_inference": opt.tiled_inference,
            "compute_eval_loss": opt.compute_eval_loss,
        },
    )

    # create a website
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    if not opt.skip_save_images:
        print(f"creating web directory {web_dir}")
    webpage = None if (opt.no_html or opt.skip_save_images) else html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")
    image_dir = web_dir / "images"
    if not opt.skip_save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    # use tiff if appropriate
    image_ext = ".tiff" if getattr(opt, "save_to_tiff", False) else ".png"
    output_imtype = np.dtype(opt.dtype).type
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    total_recon_loss = 0.0
    n_loss = 0

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        if i == 0:
            _, _, h, w = data["A"].shape
            print(f"Loaded input tensor size: {h}x{w}")
            if opt.tiled_inference:
                print(f"Effective eval patch size: {opt.tile_size}x{opt.tile_size}")
            else:
                print(f"Effective eval input size: {h}x{w}")

        if opt.tiled_inference:
            visuals, fake_B = tiled_test(model, data, opt)
        else:
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            fake_B = model.fake_B

        if opt.compute_eval_loss:
            loss = reconstruction_loss(
                pred=fake_B,
                target=data["B"].to(fake_B.device),
                opt=opt,
            )
            total_recon_loss += loss.item()
            n_loss += 1

            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"{opt.phase}/recon_loss": loss.item(),
                        f"{opt.phase}/running_avg_recon_loss": total_recon_loss / n_loss,
                    },
                    step=i,
                )

        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            if not opt.skip_save_images:
                print(f"processing ({i:04d})-th image... {img_path}")
        if not opt.skip_save_images and webpage is not None:
            save_images(
                webpage,
                visuals,
                img_path,
                aspect_ratio=opt.aspect_ratio,
                width=opt.display_winsize,
                image_ext=image_ext,
                output_imtype=output_imtype,
            )
        elif not opt.skip_save_images:
            save_visuals_to_directory(
                image_dir,
                visuals,
                img_path,
                image_ext=image_ext,
                output_imtype=output_imtype,
            )
        if getattr(opt, "wandb_log_images", False):
            log_visuals(
                wandb_run,
                visuals,
                step=i,
                prefix="eval",
                output_imtype=output_imtype,
            )

    if opt.compute_eval_loss:
        avg_recon_loss = total_recon_loss / n_loss if n_loss > 0 else float("nan")
        print(f"{opt.phase} avg_recon_loss: {avg_recon_loss:.6f}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{opt.phase}/avg_recon_loss": avg_recon_loss,
                    f"{opt.phase}/num_images": n_loss,
                }
            )

        print(f"EVAL_METRICS: {json.dumps({'avg_recon_loss': avg_recon_loss, 'num_images': n_loss}, sort_keys=True)}")
        if not opt.skip_save_images:
            metrics_path = web_dir / f"{opt.phase}_metrics.csv"
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["epoch", "num_images", "avg_recon_loss"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": opt.epoch,
                        "num_images": n_loss,
                        "avg_recon_loss": avg_recon_loss,
                    }
                )
            print(f"wrote metrics to {metrics_path}")
        update_summary(
            wandb_run,
            {
                f"{opt.phase}/avg_recon_loss": avg_recon_loss,
                f"{opt.phase}/num_images": n_loss,
            },
        )

    if webpage is not None and not opt.skip_save_images:
        webpage.save()  # save the HTML
    finish_run(wandb_run)
