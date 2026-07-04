from pathlib import Path

from util import util


def save_visuals_to_directory(output_dir, visuals, image_path, *, image_ext, output_imtype):
    name = Path(image_path[0]).stem
    for label, image in visuals.items():
        image_numpy = util.tensor2im(image, imtype=output_imtype)
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        save_path = label_dir / f"{name}{image_ext}"
        util.save_image(image_numpy, save_path)
