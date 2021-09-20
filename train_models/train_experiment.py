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
from train_models.trainer.training import Training, IncpetionTraining
from train_models.utils.data_parallel import DataParallel

Image.MAX_IMAGE_PIXELS = None


def main(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = ppl.DATASETS[args.dataset].load_dataset()

    train_ds = DownsamplePreprocess(Split.TRAIN, *dataset.get_subset(Split.TRAIN))
    val_ds = DownsamplePreprocess(Split.VAL, *dataset.get_subset(Split.VAL))

    input_pipeline = InputPipeline(
        datasets_list=[train_ds, val_ds], batch_size=args.batch_size, pin_memory=True if use_cuda else False)

    n_outputs = train_ds.get_n_outputs()
    model = ppl.ARCHITECTURE[args.architecture](num_classes=n_outputs)

    if args.pretrained:
        model_filename = args.pretrained
        model_path = os.path.join(Paths.formatted_pt_path, model_filename)
        model = load_checkpoint_pretrained(model_path, model)

    if use_cuda and torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = DataParallel(model)
    model = model.to(device)

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    saver = None
    model_path = os.path.join(Paths.training_ckpts, args.ckpt_name)
    if not args.no_ckpt:
        saver = Saver(model_path)

    training_class = IncpetionTraining if args.architecture == 'inception_v3' else Training
    training = training_class(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        saver=saver,
        device=device)

    if os.path.exists(model_path):
        training.retrain()

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    training.train(args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset.", type=DatasetArgs, choices=list(DatasetArgs))
    parser.add_argument("architecture", help="Architecture.", type=ArchArgs, choices=list(ArchArgs))
    parser.add_argument("batch_size", help="Learning rate.", type=int)
    parser.add_argument("learning_rate", help="Learning rate.", type=float)
    parser.add_argument("max_epochs", help="Number of epochs to train the model.", type=int)
    parser.add_argument("ckpt_name", help="Retrain from already existing checkpoint.", type=str)
    parser.add_argument("--no_ckpt", help="Avoid checkpointing.", default=False, action='store_true')
    parser.add_argument("--pretrained", help="Train from pretrained model.", type=str, default=False)
    args = parser.parse_args()

    assert args.dataset in ppl.DATASETS

    print(args)
    main(args)
