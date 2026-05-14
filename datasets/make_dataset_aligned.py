import os
import numpy as np
import tifffile
from PIL import Image


def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            # support numerous filetypes
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    return image_file_paths


def align_images(a_file_paths, b_file_paths, target_path):
    os.makedirs(target_path, exist_ok=True)

    for i in range(len(a_file_paths)):
        ext_a = os.path.splitext(a_file_paths[i])[1].lower()
        ext_b = os.path.splitext(b_file_paths[i])[1].lower()

        assert ext_a == ext_b, f"Extension mismatch: {a_file_paths[i]} vs {b_file_paths[i]}"

        if ext_a in (".tif", ".tiff"):
            img_a = tifffile.imread(a_file_paths[i])
            img_b = tifffile.imread(b_file_paths[i])

            assert img_a.shape == img_b.shape, f"Shape mismatch: {a_file_paths[i]} vs {b_file_paths[i]}"

            aligned_image = np.concatenate([img_a, img_b], axis=1)
            out_path = os.path.join(target_path, "{:04d}.tiff".format(i))
            tifffile.imwrite(out_path, aligned_image)

        elif ext_a in (".png", ".jpg", ".jpeg"):
            img_a = Image.open(a_file_paths[i])
            img_b = Image.open(b_file_paths[i])

            assert img_a.size == img_b.size, f"Size mismatch: {a_file_paths[i]} vs {b_file_paths[i]}"

            mode = img_a.mode
            assert img_b.mode == mode, f"Mode mismatch: {a_file_paths[i]} vs {b_file_paths[i]}"

            aligned_image = Image.new(mode, (img_a.size[0] * 2, img_a.size[1]))
            aligned_image.paste(img_a, (0, 0))
            aligned_image.paste(img_b, (img_a.size[0], 0))

            out_ext = ".png" if ext_a == ".png" else ".jpg"
            out_path = os.path.join(target_path, "{:04d}{}".format(i, out_ext))
            aligned_image.save(out_path)

        else:
            raise ValueError(f"Unsupported extension: {ext_a}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
        help='Dataset root containing trainA/trainB and testA/testB folders'
    )
    args = parser.parse_args()

    dataset_folder = args.dataset_path
    print(dataset_folder)

    ab_root = os.path.join(dataset_folder, 'AB')

    for split in ('train', 'val', 'test'):
        split_a_path = os.path.join(dataset_folder, f'{split}A')
        split_b_path = os.path.join(dataset_folder, f'{split}B')

        if not (os.path.isdir(split_a_path) and os.path.isdir(split_b_path)):
            continue

        split_a_file_paths = get_file_paths(split_a_path)
        split_b_file_paths = get_file_paths(split_b_path)
        assert(len(split_a_file_paths) == len(split_b_file_paths))

        split_path = os.path.join(ab_root, split)
        align_images(split_a_file_paths, split_b_file_paths, split_path)
