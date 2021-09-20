import argparse
import math
import os
import random
import PIL
from PIL import Image
import numpy as np
import pandas as pd
from typing import Union
from torchvision import transforms
from consts.consts import DatasetArgs, MosaicArgs
from consts.paths import MosaicPaths, DatasetPaths

def get_transformation():
    crop_size = (224, 224)
    resize_size = (int(crop_size[0] / 0.875), int(crop_size[1] / 0.875))
    transformations = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
    ])
    return transformations


def create_mosaic(mosaic_image_paths, output_path):
    transform = get_transformation()

    images = []
    for image_path in mosaic_image_paths:
        img = PIL.Image.open(image_path)
        img = img.convert('RGB')
        img = np.asarray(transform(img))
        images.append(img)

    imgs_comb = np.vstack((np.hstack((images[0], images[1])), np.hstack((images[2], images[3]))))
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(output_path)


def mosaic_creation(dataset: DatasetArgs, mosaic: MosaicArgs, mosaics_per_class: int, seed: Union[int, None] = None):
    if seed:
        random.seed(seed)

    _DatasetPath = DatasetPaths.get_from(dataset)
    _MosaicPath = MosaicPaths.get_from(mosaic)

    if not os.path.exists(_MosaicPath.images_folder):
        os.makedirs(_MosaicPath.images_folder)

    labels = pd.read_csv(_DatasetPath.label_path)['label'].to_list()
    total_labels = len(labels)
    label2idx = {label: idx for idx, label in enumerate(labels)}

    df = pd.DataFrame(columns=['filename', "filename_0", "filename_1", "filename_2", "filename_3",
                               "label_0", "label_1", "label_2", "label_3", "target_class"])
    df = df.astype({"filename": str, "filename_0": str, "filename_1": str, "filename_2": str, "filename_3": str,
                    "label_0": int, "label_1": str, "label_2": str, "label_3": str, "target_class": int})

    dataset_df = pd.read_csv(_DatasetPath.csv_path)
    dataset_df['label_idx'] = dataset_df['label'].map(label2idx)
    val_df = dataset_df[dataset_df['subset'] == 'val']

    idx_mosaic = 0

    for target_class in range(total_labels):
        outer_classes = list(range(total_labels))
        outer_classes.remove(target_class)

        target_class_df = val_df[val_df['label_idx'] == target_class]
        target_class_filenames = target_class_df['filename'].to_list()

        num_repeat = int(math.ceil((2 * mosaics_per_class) / len(target_class_filenames)))
        total_target_class_filenames = num_repeat * target_class_filenames
        random.shuffle(total_target_class_filenames)

        repeat_labels = int(math.ceil((2 * mosaics_per_class) / len(outer_classes)))
        total_outer_filenames = []
        total_outer_classes = []
        for outer_class in outer_classes:
            outer_class_df = val_df[val_df['label_idx'] == outer_class]
            total_outer_filenames += outer_class_df['filename'].sample(repeat_labels, random_state=seed if seed else None).to_list()
            total_outer_classes += [outer_class for _ in range(repeat_labels)]
        total_outer = list(zip(total_outer_filenames, total_outer_classes))
        random.shuffle(total_outer)

        iter_filenames = iter(total_target_class_filenames)
        iter_outer = iter(total_outer)
        for _ in range(mosaics_per_class):
            mosaic_elems = [
                (next(iter_filenames), target_class), (next(iter_filenames), target_class),
                next(iter_outer), next(iter_outer)
            ]
            random.shuffle(mosaic_elems)
            mosaic_name = f'{idx_mosaic}.jpg'
            new_row = {"filename": mosaic_name, "target_class": target_class}
            for idx, (_filename, _class) in enumerate(mosaic_elems):
                new_row[f'filename_{idx}'] = _filename
                new_row[f'label_{idx}'] = _class
            df = df.append(new_row, ignore_index=True)

            mosaic_image_paths = [os.path.join(_DatasetPath.images_folder, _filename) for _filename, _class in
                                  mosaic_elems]
            mosaic_path = os.path.join(_MosaicPath.images_folder, mosaic_name)
            create_mosaic(mosaic_image_paths, mosaic_path)

            idx_mosaic += 1

    df.to_csv(_MosaicPath.csv_path, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name", type=DatasetArgs, choices=list(DatasetArgs))
    parser.add_argument("mosaic", help="Mosaic name", type=MosaicArgs, choices=list(MosaicArgs))
    parser.add_argument("--seed", help="Random seed", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    mosaic_creation(args.dataset, args.mosaic, 100, seed=args.seed)
