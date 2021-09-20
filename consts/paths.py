import os
import hashlib

from consts.consts import DatasetArgs, MosaicArgs, XmethodArgs, ArchArgs

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir,) * 2))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
CKPTS_PATH = os.path.join(PROJECT_PATH, 'checkpoints')


class Paths:
    summaries_folder = os.path.join(PROJECT_PATH, 'train_models', 'summaries')
    raw_pt_path = os.path.join(CKPTS_PATH, 'raw_pt_ckpts')
    formatted_pt_path = os.path.join(CKPTS_PATH, 'formatted_pt_ckpts')
    training_ckpts = os.path.join(CKPTS_PATH, 'training_ckpts')

    datasets_path = os.path.join(DATA_PATH, 'datasets')
    mosaics_path = os.path.join(DATA_PATH, 'mosaics')
    explainability_path = os.path.join(DATA_PATH, 'explainability')
    explainability_csv = os.path.join(explainability_path, "hash_explainability.csv")
    evaluation_path = os.path.join(DATA_PATH, 'evaluation')
    logits_path = os.path.join(DATA_PATH, 'logits')
    relevance_results = os.path.join(explainability_path, 'relevances')
    metric_results = os.path.join(explainability_path, 'metrics')


class DatasetPaths:
    @classmethod
    def get_from(cls, dataset: DatasetArgs):
        dataset_map = {
            DatasetArgs.CATSDOGS: cls.CatsDogs,
            DatasetArgs.MIT67: cls.Mit67,
            DatasetArgs.ILSVRC2012: cls.Ilsvrc2012,
            DatasetArgs.MAME: cls.Mame
        }
        return dataset_map[dataset]

    class CatsDogs:
        images_folder = os.path.join(Paths.datasets_path, 'catsdogs', 'data')
        csv_path = os.path.join(Paths.datasets_path, 'catsdogs', 'catsdogs.csv')
        label_path = os.path.join(Paths.datasets_path, 'catsdogs', 'catsdogs_labels.csv')

    class Mit67:
        images_folder = os.path.join(Paths.datasets_path, 'mit67', 'data')
        csv_path = os.path.join(Paths.datasets_path, 'mit67', 'mit67.csv')
        label_path = os.path.join(Paths.datasets_path, 'mit67', 'mit67_labels.csv')

    class Ilsvrc2012:
        images_folder = os.path.join(Paths.datasets_path, 'ilsvrc2012', 'data')
        csv_path = os.path.join(Paths.datasets_path, 'ilsvrc2012', 'ilsvrc2012.csv')
        label_path = os.path.join(Paths.datasets_path, 'ilsvrc2012', 'ilsvrc2012_labels.csv')

    class Mame:
        images_folder = os.path.join(Paths.datasets_path, 'mame', 'data')
        csv_path = os.path.join(Paths.datasets_path, 'mame', 'mame.csv')
        label_path = os.path.join(Paths.datasets_path, 'mame', 'mame_labels.csv')



class MosaicPaths:
    @classmethod
    def get_from(cls, dataset: MosaicArgs):
        dataset_map = {
            MosaicArgs.CATSDOGS_MOSAIC: cls.CatsDogsMosaic,
            MosaicArgs.MIT67_MOSAIC: cls.Mit67Mosaic,
            MosaicArgs.ILSVRC2012_MOSAIC: cls.Ilsvrc2012Mosaic,
            MosaicArgs.MAME_MOSAIC: cls.MameMosaic,
        }
        return dataset_map[dataset]

    class CatsDogsMosaic:
        images_folder = os.path.join(Paths.mosaics_path, 'catsdogs_mosaic', 'data')
        csv_path = os.path.join(Paths.mosaics_path, 'catsdogs_mosaic', 'catsdogs.csv')

    class Mit67Mosaic:
        images_folder = os.path.join(Paths.mosaics_path, 'mit67_mosaic', 'data')
        csv_path = os.path.join(Paths.mosaics_path, 'mit67_mosaic', 'mit67.csv')

    class Ilsvrc2012Mosaic:
        images_folder = os.path.join(Paths.mosaics_path, 'ilsvrc2012_mosaic', 'data')
        csv_path = os.path.join(Paths.mosaics_path, 'ilsvrc2012_mosaic', 'ilsvrc.csv')

    class MameMosaic:
        images_folder = os.path.join(Paths.mosaics_path, 'mame_mosaic', 'data')
        csv_path = os.path.join(Paths.mosaics_path, 'mame_mosaic', 'mame.csv')

def get_heatmaps_folder(xmethod: XmethodArgs, dataset: DatasetArgs, architecture: ArchArgs, ckpt: str):
    string2hash = f"{xmethod}{dataset}{architecture}{ckpt}"
    _hash = hashlib.md5(string2hash.encode('utf-8')).hexdigest()
    return _hash, os.path.join(Paths.explainability_path, f"{_hash}")
