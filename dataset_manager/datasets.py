import os

import pandas as pd

from consts.consts import Split
from consts.paths import DatasetPaths
from consts.paths import MosaicPaths


class BasicDataset:
    def __init__(self, csv_path, data_folder):
        self.data_folder = data_folder
        self.df = pd.read_csv(csv_path)

    def get_subset(self, subset):
        assert subset in Split
        df_subset = self.df[self.df['subset'] == subset.value]
        image_filenames = df_subset['filename'].tolist()
        image_labels = df_subset['label'].tolist()
        image_filepaths = self.get_filepaths(image_filenames, image_labels, subset)
        return image_filepaths, image_labels

    def get_n_outputs(self):
        _, image_labels = self.get_subset(Split.TRAIN)
        unique_labels = len(set(image_labels))
        return unique_labels

    def get_filepaths(self, image_filenames, image_labels, subset):
        return [os.path.join(self.data_folder, img_filename) for img_filename in image_filenames]

    @classmethod
    def load_dataset(cls):
        raise NotImplementedError


class CatsDogsDataset(BasicDataset):

    @classmethod
    def load_dataset(cls):
        csv_path = DatasetPaths.CatsDogs.csv_path
        data_folder = DatasetPaths.CatsDogs.images_folder
        return cls(csv_path, data_folder)


class Mit67Dataset(BasicDataset):

    @classmethod
    def load_dataset(cls):
        csv_path = DatasetPaths.Mit67.csv_path
        data_folder = DatasetPaths.Mit67.images_folder
        return cls(csv_path, data_folder)


class Ilsvrc2012Dataset(BasicDataset):

    @classmethod
    def load_dataset(cls):
        csv_path = DatasetPaths.Ilsvrc2012.csv_path
        data_folder = DatasetPaths.Ilsvrc2012.images_folder
        return cls(csv_path, data_folder)


class MameDataset(BasicDataset):

    @classmethod
    def load_dataset(cls):
        csv_path = DatasetPaths.Mame.csv_path
        data_folder = DatasetPaths.Mame.images_folder
        return cls(csv_path, data_folder)


class MosaicDataset(BasicDataset):
    def get_subset(self, subset):
        assert subset in Split

        list_image_filenames, list_image_labels = [], []
        for index, row in self.df.iterrows():
            mosaic_filenames = [row.loc[f'filename_{i}'] for i in range(4)]
            mosaic_labels = [row.loc[f'label_{i}'] for i in range(4)]

            list_image_filenames.append(mosaic_filenames)
            list_image_labels.append(mosaic_labels)

        mosaic_filenames = self.df['filename'].tolist()
        mosaic_filepaths = self.get_filepaths(mosaic_filenames, None, None)
        target_classes = self.df['target_class'].tolist()

        return mosaic_filepaths, target_classes, list_image_filenames, list_image_labels

    def get_n_outputs(self):
        _, target_classes, _, _ = self.get_subset(Split.TRAIN)
        unique_labels = len(set(target_classes))
        return unique_labels


class Mit67Mosaic(MosaicDataset):
    @classmethod
    def load_dataset(cls):
        csv_path = MosaicPaths.Mit67Mosaic.csv_path
        data_folder = MosaicPaths.Mit67Mosaic.images_folder
        return cls(csv_path, data_folder)


class CatsDogsMosaic(MosaicDataset):
    @classmethod
    def load_dataset(cls):
        csv_path = MosaicPaths.CatsDogsMosaic.csv_path
        data_folder = MosaicPaths.CatsDogsMosaic.images_folder
        return cls(csv_path, data_folder)


class Ilsvrc2012Mosaic(MosaicDataset):
    @classmethod
    def load_dataset(cls):
        csv_path = MosaicPaths.Ilsvrc2012Mosaic.csv_path
        data_folder = MosaicPaths.Ilsvrc2012Mosaic.images_folder
        return cls(csv_path, data_folder)


class MameMosaic(MosaicDataset):
    @classmethod
    def load_dataset(cls):
        csv_path = MosaicPaths.MameMosaic.csv_path
        data_folder = MosaicPaths.MameMosaic.images_folder
        return cls(csv_path, data_folder)
