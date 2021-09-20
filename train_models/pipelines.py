from models.resnet import resnet18
from models.vgg import vgg16
from models.alexnet import alexnet
from dataset_manager.datasets import Mit67Dataset, CatsDogsDataset, Ilsvrc2012Dataset, MameDataset
from consts.consts import DatasetArgs, ArchArgs

DATASETS = {
    DatasetArgs.CATSDOGS: CatsDogsDataset,
    DatasetArgs.MIT67: Mit67Dataset,
    DatasetArgs.ILSVRC2012: Ilsvrc2012Dataset,
    DatasetArgs.MAME: MameDataset,
}

ARCHITECTURE = {
    ArchArgs.VGG16: vgg16,
    ArchArgs.RESNET18: resnet18,
    ArchArgs.ALEXNET: alexnet,
}
