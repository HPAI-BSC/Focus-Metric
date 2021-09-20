import os
import argparse

import numpy as np
import pandas as pd
import torch
import wget
import zipfile

from consts.paths import get_heatmaps_folder, Paths, MosaicPaths
from consts.consts import ArchArgs, XmethodArgs, Split, MosaicArgs
from train_models.trainer.saver import load_checkpoint_pretrained, load_checkpoint
from explainability import pipelines as ppl
from explainability.utils import use_cuda, create_dir




def load_model(arch_args: str, ckpt: str, n_outputs: int):
    model = ppl.ARCHITECTURE[arch_args](num_classes=n_outputs)
    ckpt_path = os.path.join(Paths.formatted_pt_path, ckpt)
    if not os.path.exists(ckpt_path):
        if not os.path.exists(Paths.formatted_pt_path):
            os.makedirs(Paths.formatted_pt_path)
        print(f'https://storage.hpai.bsc.es/focus-metric/formatted_pt_checkpoints/{ckpt}')
        print(f'Donwloading {ckpt} ...')
        try:
            wget.download(url=f'https://storage.hpai.bsc.es/focus-metric/formatted_pt_checkpoints/{ckpt}', out=Paths.formatted_pt_path)
            print("\nDone!")
        except:
            print("Error downloading checkpoints!")
    model, _, _ = load_checkpoint(ckpt_path, model, None)
    device = torch.device("cuda" if use_cuda() else "cpu")
    model = model.to(device)
    model.eval()
    return model



def main(args):

    if not os.path.exists(MosaicPaths.get_from(args.dataset).images_folder):
        if not os.path.exists(Paths.mosaics_path):
            os.makedirs(Paths.mosaics_path)
        print(f'Donwloading {args.dataset} dataset ...')
        try:
            wget.download(url=f'https://storage.hpai.bsc.es/focus-metric/{args.dataset}.zip', out=Paths.mosaics_path)
            zip_path = os.path.join(Paths.mosaics_path, f'{args.dataset}.zip')
            with zipfile.ZipFile(zip_path, 'r') as h:
                h.extractall(Paths.mosaics_path)
            os.remove(zip_path)
            print("\nDone!")
        except:
            print("Error downloading mosaics!")


    dataset = ppl.DATASETS[args.dataset].load_dataset()
    model = load_model(args.architecture, args.ckpt, dataset.get_n_outputs())
    explaining_method = ppl.XMETHOD[args.Xmethod](model)

    _hash, heatmaps_path = get_heatmaps_folder(args.Xmethod, args.dataset, args.architecture, args.ckpt)
    create_dir(heatmaps_path)
    try:
        hash_df = pd.read_csv(Paths.explainability_csv)
    except FileNotFoundError:
        hash_df = pd.DataFrame(columns=["hash", "dataset", "architecture", "checkpoint", "xmethod"])

    if len(hash_df[hash_df['hash'] == _hash]) == 0:
        new_row = {
            "dataset": args.dataset,
            "architecture": args.architecture,
            "checkpoint": args.ckpt,
            "xmethod": args.Xmethod,
            "hash": _hash
        }
        hash_df = hash_df.append(new_row, ignore_index=True)
        hash_df.to_csv(Paths.explainability_csv, index=False)

    for mosaic_filepath, target_class, images_filenames, image_labels in zip(*dataset.get_subset(Split.VAL)):
        mosaic_name = os.path.splitext(os.path.basename(mosaic_filepath))[0]
        output_path = os.path.join(heatmaps_path, f"{mosaic_name}.npy")

        if not os.path.exists(output_path):
            mosaic_explanation = explaining_method.explain(mosaic_filepath, target_class)
            np.save(output_path, mosaic_explanation)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset.", type=MosaicArgs, choices=list(MosaicArgs))
    parser.add_argument("architecture", help="Architecture.", type=ArchArgs, choices=list(ArchArgs))
    parser.add_argument("ckpt", type=str, help="Model checkpoint")
    parser.add_argument('Xmethod', type=XmethodArgs, choices=XmethodArgs, help='Explainability methods')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
