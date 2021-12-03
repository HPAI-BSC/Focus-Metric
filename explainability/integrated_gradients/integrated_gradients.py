import torch
import numpy as np
import matplotlib.cm
from torch.autograd import Variable

from explainability.explainability_class import Explainability
from explainability.integrated_gradients.integrated_gradients_original import IntegratedGradients as IntGrad


class IntegratedGradients(Explainability):
    def __init__(self, model):
        self.model = model
        self.ig = IntGrad(self.model)
        super().__init__('integratedgradients')

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        torch.cuda.empty_cache()
        cuda = torch.cuda.is_available()
        img = img.cuda() if cuda else img
        img = Variable(img, volatile=False, requires_grad=True)
        heatmap, delta = self.ig.attribute(img, target=target_class, return_convergence_delta=True)
        heatmap = np.transpose(heatmap.squeeze().cpu().detach().numpy(), (1, 2, 0))
        heatmap = np.sum(heatmap, axis=2)
        return heatmap

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
        alpha = 0.2
        out_img[:,:,:] = (alpha * img[:,:,:]) + ((1-alpha) * rgb[:,:,:])
        return out_img
