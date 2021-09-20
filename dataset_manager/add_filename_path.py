import os
import argparse
import pandas as pd
from consts.paths import Paths

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Dataset name", type=str)
    args = parser.parse_args()
    return args


def add_filename_path(args):
    dataset_path = os.path.join(Paths.datasets_path, args.dataset_name, f'{args.dataset_name}.csv')
    dataset_df = pd.read_csv(dataset_path)

    filenames_paths = [os.path.join(row['subset'], row['label'], row['filename']) for index, row in dataset_df.iterrows()]
    print(filenames_paths)
    dataset_df['filename_path'] = filenames_paths
    dataset_df.to_csv(dataset_path, index=False)


if __name__ == '__main__':
    args = get_args()
    add_filename_path(args)
