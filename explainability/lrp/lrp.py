import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.cm

from explainability.explainability_class import Explainability


class LRP(Explainability):    
    def __init__(self, model):
        self.model = model.to('cpu')
        super().__init__('lrp')

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        heatmap_array = self.compute_heatmap(img, target_class)
        return heatmap_array[0, :, :, 0]

    @staticmethod
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

    @staticmethod
    def get_scores(_tensor):
        x = torch.sigmoid(_tensor)
        scores = np.array(x.data.view(-1))
        return scores

    def compute_heatmap(self, img, target_category):
        input = Variable(img, volatile=True).cpu()
        input.requires_grad = True
        output = self.model(input)
        scores = self.get_scores(output)
        n_outputs = len(scores)
        _tensor = self.compute_tensor(output, n_outputs, target_category)
        res = self.model.relprop(relevance=output * _tensor, alpha=1).sum(dim=1, keepdim=True)
        heatmap_array = res.permute(0, 2, 3, 1).data.cpu().numpy()
        return heatmap_array
    
    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        cmap_name = 'seismic'
        cmap = eval(f'matplotlib.cm.{cmap_name}')
        heatmap = heatmap / np.max(np.abs(heatmap))  # normalize to [-1,1] wrt to max heatmap magnitude
        heatmap = (heatmap + 1.) / 2.  # shift/normalize to [0,1] for color mapping
        rgb = cmap(heatmap.flatten())[..., 0:3].reshape([heatmap.shape[0], heatmap.shape[1], 3])
        img = img.numpy()[0, :, :, :]  # one image
        img = np.moveaxis(img, 0, 2)  # change color channel position
        img = np.float32(img)
        out_img = np.zeros(img.shape,dtype=img.dtype)
        alpha = 0.05
        out_img[:,:,:] = (alpha * img[:,:,:]) + ((1-alpha) * rgb[:,:,:])
        return out_img
