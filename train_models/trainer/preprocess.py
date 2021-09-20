from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from consts.consts import Split


class BasePreprocess(Dataset):

    def __init__(self, subset, image_paths, image_labels, order=False, target_class=False, filename=False):
        assert subset not in [attr for attr in dir(Split) if not attr.startswith('__')]
        if order: assert isinstance(order, list)
        if target_class: assert isinstance(target_class, list)
        if filename: assert isinstance(filename, list)
        self.subset = subset
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.order = order
        self.target_class = target_class
        self.filename = filename
        self.transform = transforms.Compose(self._get_transforms_list())
        self.label_transformation = None

        self.n_outputs = len(set(image_labels))
        self.set_up_label_transformation_for_classification()

    def set_up_label_transformation_for_classification(self):
        sorted_labels = sorted(set(self.image_labels))
        label2idx = {raw_label: idx for idx, raw_label in enumerate(sorted_labels)}
        self.label_transformation = lambda x: torch.tensor(label2idx[x], dtype=torch.long).view(-1, 1)

    def _get_transforms_list(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

    def get_n_outputs(self):
        return self.n_outputs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        label = self.image_labels[idx]
        if self.label_transformation:
            label = self.label_transformation(label)
        image = self.transform(image)

        _return = [image, label]
        if self.order:
            _return.append(self.order[idx])
        if self.target_class:
            _return.append(self.target_class[idx])
        if self.filename:
            _return.append(self.filename[idx])

        return _return


class DownsamplePreprocess(BasePreprocess):
    def __init__(self, *args, **kwargs):
        try:
            size = kwargs['size']
            del kwargs['size']
        except KeyError:
            size = (224, 224)
        self.crop_size = size
        self.resize_size = (int(self.crop_size[0] / 0.875), int(self.crop_size[1] / 0.875))
        super().__init__(*args, **kwargs)

    def _get_transforms_list(self):
        if self.subset == Split.TRAIN:
            return_transform = [
                transforms.Resize(self.resize_size),
                transforms.RandomRotation(degrees=30),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        else:
            return_transform = [
                transforms.Resize(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        return return_transform
