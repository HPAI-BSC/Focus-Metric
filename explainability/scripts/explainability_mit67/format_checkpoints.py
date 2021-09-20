import argparse
import collections
import copy
import os

import torch

from consts.consts import ArchArgs
from consts.paths import Paths
from models.alexnet import alexnet
from checkpoint_manager.raw_checkpoints import download_raw_ckpt
from models.resnet import resnet18
from models.vgg import vgg16


def base_loading(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model_dict = model.state_dict()
    for k, v in checkpoint.items():
        cpt = {}
        cpt[k] = v
        try:
            model_dict.update(cpt)
            model.load_state_dict(model_dict)
        except RuntimeError:
            pass
    return model


def places_loading(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        k = str.replace(k, 'module.', '')
        for l, p in model.named_parameters():
            if l in k:
                cpt = {}
                cpt[k] = v
                try:
                    model_dict.update(cpt)
                    model.load_state_dict(model_dict)
                except RuntimeError:
                    pass
    return model


def places_vgg_loading(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model_dict = model.state_dict()
    list_names = []
    [list_names.append(layer_name) for layer_name, _ in model.named_parameters()]
    model_aux = copy.deepcopy(model)
    model_aux.features = torch.nn.Sequential(collections.OrderedDict(
        zip(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
             'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
             'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
             'pool5'], model.features)))
    model_aux.classifier = torch.nn.Sequential(
        collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8a'], model.classifier)))
    for k, v in checkpoint.items():
        idx = 0
        for l, p in model_aux.named_parameters():
            if k in l:
                cpt = {}
                cpt[list_names[idx]] = v
                try:
                    model_dict.update(cpt)
                    model.load_state_dict(model_dict)
                except RuntimeError:
                    pass
            idx += 1
    return model


def load_checkpoint_pretrained(model_path, model, loading_func):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if not os.path.isfile(model_path):
        print("=> no checkpoint found at '{}'".format(model_path))
        return model

    print("=> loading checkpoint '{}'".format(model_path))
    model = loading_func(model, model_path)
    print("=> loaded checkpoint '{}'".format(model_path))

    return model


ARCH = "architecture"
CKPT = "checkpoint"
LOADING = "loading_method"
CLASSES = "classes"

ARCHITECTURE = {
    ArchArgs.VGG16: vgg16,
    ArchArgs.RESNET18: resnet18,
    ArchArgs.ALEXNET: alexnet,
}

ckpt_to_args = {
    "alexnet-owt-4df8aa71.pth": {
        ARCH: ArchArgs.ALEXNET,
        CKPT: "alexnet_pt_imagenet.ckpt",
        LOADING: base_loading,
        CLASSES: 1000},
    "alexnet_places365.pth.tar": {
        ARCH: ArchArgs.ALEXNET,
        CKPT: "alexnet_pt_places.ckpt",
        LOADING: places_loading,
        CLASSES: 365},
    "resnet18-5c106cde.pth": {
        ARCH: ArchArgs.RESNET18,
        CKPT: "resnet18_pt_imagenet.ckpt",
        LOADING: base_loading,
        CLASSES: 1000},
    "resnet18_places365.pth.tar": {
        ARCH: ArchArgs.RESNET18,
        CKPT: "resnet18_pt_places.ckpt",
        LOADING: places_loading,
        CLASSES: 365},
    "vgg16-397923af.pth": {
        ARCH: ArchArgs.VGG16,
        CKPT: "vgg16_pt_imagenet.ckpt",
        LOADING: base_loading,
        CLASSES: 1000},
    "vgg16_places365.caffemodel.pt": {
        ARCH: ArchArgs.VGG16,
        CKPT: "vgg16_pt_places.ckpt",
        LOADING: places_vgg_loading,
        CLASSES: 365}
}


def main(unformated_ckpt: str):
    print(ckpt_to_args[args.unformatted_ckpt])

    ckpt_input_filename = os.path.join(Paths.raw_pt_path, unformated_ckpt)
    ckpt_output_filename = os.path.join(Paths.formatted_pt_path, ckpt_to_args[unformated_ckpt][CKPT])
    if os.path.exists(ckpt_output_filename):
        print(f"{unformated_ckpt} has been already formated -> {ckpt_output_filename}")
        return
    else:
        if not os.path.exists(Paths.formatted_pt_path):
            os.makedirs(Paths.formatted_pt_path)

    if not os.path.exists(ckpt_input_filename):
        if not os.path.exists(Paths.raw_pt_path):
            os.makedirs(Paths.raw_pt_path)
        download_raw_ckpt(ckpt_input_filename)

    model = ARCHITECTURE[ckpt_to_args[unformated_ckpt][ARCH]](num_classes=ckpt_to_args[args.unformatted_ckpt][CLASSES])
    model = load_checkpoint_pretrained(ckpt_input_filename, model, ckpt_to_args[unformated_ckpt][LOADING])

    torch.save({'state_dict': model.state_dict()}, ckpt_output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unformatted_ckpt", help="Name of the unformatted checkpoint to format.", type=str)
    args = parser.parse_args()

    main(args.unformatted_ckpt)
