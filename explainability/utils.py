import os
import torch
import cv2
import imageio

from torch import nn
import numpy as np
import pandas as pd
from numpy import save

# ------------------------------------------------------
# Create directory if it does not exist
# ------------------------------------------------------
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def use_cuda():
    return torch.cuda.is_available()

def model_randomization(m):
    print('m', m)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)


def extract_mosaic(hmap):
    size = hmap.shape

    quadrant_0 = hmap[0:int(size[0] / 2), 0:int(size[1] / 2)]
    quadrant_1 = hmap[0:int(size[0] / 2), int(size[1] / 2):size[1]]
    quadrant_2 = hmap[int(size[0] / 2):size[0], 0:int(size[1] / 2)]
    quadrant_3 = hmap[int(size[0] / 2):size[0], int(size[1] / 2):size[1]]

    mosaic = [quadrant_0, quadrant_1, quadrant_2, quadrant_3]
    return mosaic


def extract_mosaic_noise(hmap):
    size = hmap.shape
    margin = hmap[0:int(size[0] / 2)].shape[0] / 8
    quadrant_0 = hmap[0 + int(margin):int(size[0] / 2) - int(margin), 0 + int(margin):int(size[1] / 2) - int(margin)]
    quadrant_1 = hmap[0 + int(margin):int(size[0] / 2) - int(margin),
                 int(size[1] / 2) + int(margin):size[1] - int(margin)]
    quadrant_2 = hmap[int(size[0] / 2) + int(margin):size[0] - int(margin),
                 0 + int(margin):int(size[1] / 2) - int(margin)]
    quadrant_3 = hmap[int(size[0] / 2) + int(margin):size[0] - int(margin),
                 int(size[1] / 2) + int(margin):size[1] - int(margin)]

    mosaic = [quadrant_0, quadrant_1, quadrant_2, quadrant_3]
    return mosaic


def sum_mosaics_margin(mosaics):
    sum = 0.0
    for i in range(0, 4):
        mosaic = mosaics[i]
        sum += np.sum(mosaic[mosaic > 0])
    return sum


def compute_rel_distribution(hmap, target_category, order):
    sum_hmap = np.sum(hmap[hmap > 0])
    mosaic = extract_mosaic(hmap)
    idx_target = np.where(np.array(order) == target_category)[0]
    target_relevance = np.sum(
        [np.sum(mosaic[idx_target[i]][mosaic[idx_target[i]] > 0]) for i in range(0, len(idx_target))])
    positive_relevance = (np.sum(target_relevance) / sum_hmap) * 100
    print('positive_relevance: ', positive_relevance)
    number_of_positive_pixels = np.sum(hmap > 0)
    return positive_relevance, number_of_positive_pixels


def save_rel_distribution(rel_left, num_px_rel, target_category, idx, dataset, relevance_path):
    create_dir(relevance_path)
    csv_path = os.path.join(relevance_path, f'{dataset}.csv')
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['filename', 'target_category', 'positive_rel', 'num_px_rel'])

    exists = df.loc[df['filename'] == f'{idx}.jpg']
    if exists.empty:
        new_row = {"filename": f'{idx}.jpg', "target_category": target_category, "positive_rel": rel_left,
                   "num_px_rel": num_px_rel}
        df = df.append(new_row, ignore_index=True)
        df.to_csv(csv_path, index=False)


def save_map(map, img, filename, method, results_path):
    create_dir(results_path)
    if str(method) == 'lrp':
        imageio.imsave(os.path.join(results_path, f'{filename}.jpg'), img, vmax=1, vmin=-1)
    else:
        cv2.imwrite(os.path.join(results_path, f'{filename}.jpg'), img)
    create_dir(results_path + '_array')
    save(os.path.join(results_path + '_array', f'{filename}.npy'), map)