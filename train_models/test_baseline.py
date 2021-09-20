import os
import argparse

import torch
from PIL import Image

from consts.paths import Paths
from consts.consts import Split
from train_models.trainer.saver import Saver, load_checkpoint_pretrained
from consts.consts import DatasetArgs, ArchArgs
from train_models import pipelines as ppl
from train_models.trainer.preprocess import DownsamplePreprocess
from train_models.trainer.input import InputPipeline
from train_models.trainer.testing import Testing, InceptionTesting
from train_models.utils.data_parallel import DataParallel

Image.MAX_IMAGE_PIXELS = None


def main(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = ppl.DATASETS[args.dataset].load_dataset()

    test_ds = DownsamplePreprocess(Split.VAL, *dataset.get_subset(Split.VAL))

    input_pipeline = InputPipeline(
        datasets_list=[test_ds], batch_size=args.batch_size, pin_memory=True if use_cuda else False)

    n_outputs = test_ds.get_n_outputs()
    model = ppl.ARCHITECTURE[args.architecture](num_classes=n_outputs)
    model = model.to(device)

    if use_cuda and torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = DataParallel(model)

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    testing_class = InceptionTesting if args.architecture == 'inception_v3' else Testing
    testing = testing_class(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        device=device)

    model_path = os.path.join(Paths.formatted_pt_path, args.ckpt_name)
    if os.path.exists(model_path):
        testing.load(model_path)

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    testing.test_epoch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset.", type=DatasetArgs, choices=list(DatasetArgs))
    parser.add_argument("architecture", help="Architecture.", type=ArchArgs, choices=list(ArchArgs))
    parser.add_argument("batch_size", help="Learning rate.", type=int)
    parser.add_argument("ckpt_name", help="Retrain from already existing checkpoint.", type=str)
    args = parser.parse_args()

    assert args.dataset in ppl.DATASETS

    print(args)
    main(args)
