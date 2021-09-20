import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.cm


def compute_tensor(output, n_outputs, target_category):
    if target_category is None:
        _tensor = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        _tensor = _tensor.squeeze().cpu().numpy()

    else:
        _tensor = target_category

    _tensor = np.expand_dims(_tensor, 0)
    _tensor = (_tensor[:, np.newaxis] == np.arange(n_outputs)) * 1.0
    _tensor = torch.from_numpy(_tensor).type(torch.FloatTensor)
    return Variable(_tensor).cpu()


def hm_to_rgb(relevance, cmap='bwr', normalize=True):
    cmap = eval('matplotlib.cm.{}'.format(cmap))
    if normalize:
        relevance = relevance / np.max(np.abs(relevance))  # normalize to [-1,1] wrt to max relevance magnitude
        relevance = (relevance + 1.) / 2.  # shift/normalize to [0,1] for color mapping
    relevance = relevance
    rgb = cmap(relevance.flatten())[..., 0:3].reshape([relevance.shape[1], relevance.shape[2], 3])
    return rgb


def get_scores(_tensor):
    x = torch.sigmoid(_tensor)
    scores = np.array(x.data.view(-1))
    return scores


def compute_heatmap(img, model, target_category):
    input = Variable(img, volatile=True).cpu()
    input.requires_grad = True
    output = model(input)
    scores = get_scores(output)
    n_outputs = len(scores)  # revisar
    _tensor = compute_tensor(output, n_outputs, target_category)
    res = model.relprop(relevance=output * _tensor, alpha=1).sum(dim=1, keepdim=True)
    heatmap_array = res.permute(0, 2, 3, 1).data.cpu().numpy()
    return heatmap_array
