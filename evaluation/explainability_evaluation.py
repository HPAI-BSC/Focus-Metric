import argparse
import os

import numpy as np
import pandas as pd

from consts.consts import Split, MosaicArgs
from consts.paths import Paths
from evaluation.focus_metric import compute_focus
from explainability import pipelines as ppl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hash", help="Hash corresponding to an explainability experiment", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    df = pd.DataFrame(columns=['filename', 'focus'])

    heatmaps_path = os.path.join(Paths.explainability_path, args.hash)
    hash_df = pd.read_csv(Paths.explainability_csv)
    explainability_experiment = hash_df[hash_df['hash'] == args.hash]
    mosaic_name = explainability_experiment['dataset'].to_list()[0]
    mosaic_arg = MosaicArgs(mosaic_name)

    dataset = ppl.DATASETS[mosaic_arg].load_dataset()

    for mosaic_filepath, target_class, images_filenames, image_labels in zip(*dataset.get_subset(Split.VAL)):
        mosaic_name = os.path.splitext(os.path.basename(mosaic_filepath))[0]
        hmap = np.load(os.path.join(heatmaps_path, f'{mosaic_name}.npy'))
        focus = compute_focus(hmap, target_class, image_labels)

        new_row = {"filename": mosaic_name, "focus": focus}
        df = df.append(new_row, ignore_index=True)

    print("Median focus:", df["focus"].median())
    print("Variance focus:", df["focus"].var())
    df.to_csv(f"{heatmaps_path}.csv", index=False)
