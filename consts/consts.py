import enum


class EnumConstant(enum.Enum):
    def __str__(self):
        return self.value


class Split(EnumConstant):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class DatasetArgs(EnumConstant):
    CATSDOGS = 'catsdogs'
    MIT67 = 'mit67'
    ILSVRC2012 = 'ilsvrc2012'
    MAME = 'mame'


class MosaicArgs(EnumConstant):
    CATSDOGS_MOSAIC = 'catsdogs_mosaic'
    MIT67_MOSAIC = 'mit67_mosaic'
    ILSVRC2012_MOSAIC = 'ilsvrc2012_mosaic'
    MAME_MOSAIC ='mame_mosaic'


class XmethodArgs(EnumConstant):
    LRP = 'lrp'
    GRADCAM = 'gradcam'
    GRADCAMPLUSPLUS = 'gradcam++'
    SMOOTHGRAD = 'smoothgrad'
    INTGRAD = 'intgrad'
    LIME = 'lime'

class ArchArgs(EnumConstant):
    RESNET18 = 'resnet18'
    VGG16 = 'vgg16'
    ALEXNET = 'alexnet'
