from models.resnet import resnet18
from models.vgg import vgg16
from models.alexnet import alexnet
from dataset_manager.datasets import CatsDogsMosaic, Mit67Mosaic, Ilsvrc2012Mosaic, MameMosaic
from consts.consts import MosaicArgs, ArchArgs, XmethodArgs
from explainability.lrp.lrp import LRP
from explainability.gradcam.gradcam import Gradcam
from explainability.gradients.smoothgrad import SmoothGrad

DATASETS = {
    MosaicArgs.CATSDOGS_MOSAIC: CatsDogsMosaic,
    MosaicArgs.MIT67_MOSAIC: Mit67Mosaic,
    MosaicArgs.ILSVRC2012_MOSAIC: Ilsvrc2012Mosaic,
    MosaicArgs.MAME_MOSAIC: MameMosaic
}

ARCHITECTURE = {
    ArchArgs.VGG16: vgg16,
    ArchArgs.RESNET18: resnet18,
    ArchArgs.ALEXNET: alexnet
}

XMETHOD = {
    XmethodArgs.LRP: LRP,
    XmethodArgs.GRADCAM: Gradcam,
    XmethodArgs.SMOOTHGRAD: SmoothGrad,
}