import numpy as np


def extract_mosaic_relevance(heatmap):
    heatmap_width, heatmap_height = heatmap.shape
    quadrant_0 = heatmap[0:int(heatmap_width / 2), 0:int(heatmap_height / 2)]
    quadrant_1 = heatmap[0:int(heatmap_width / 2), int(heatmap_height / 2):heatmap_height]
    quadrant_2 = heatmap[int(heatmap_width / 2):heatmap_width, 0:int(heatmap_height / 2)]
    quadrant_3 = heatmap[int(heatmap_width / 2):heatmap_width, int(heatmap_height / 2):heatmap_height]
    relevance_per_quadrant = [quadrant_0, quadrant_1, quadrant_2, quadrant_3]
    return relevance_per_quadrant


def compute_focus(heatmap, target_category, order):
    total_relevance = np.sum(heatmap[heatmap > 0])
    relevance_per_quadrant = extract_mosaic_relevance(heatmap)
    idx_target = np.where(np.array(order) == target_category)[0]
    target_relevance = np.sum(
        [np.sum(relevance_per_quadrant[idx_target[i]][relevance_per_quadrant[idx_target[i]] > 0]) for i in
         range(len(idx_target))])
    correct_focus = np.sum(target_relevance) / total_relevance
    return correct_focus
