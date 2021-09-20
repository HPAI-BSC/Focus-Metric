import numpy as np
import torch
from torch.autograd import Variable


from explainability.explainability_class import Explainability
from explainability.gradients.smoothgrad_original import SmoothGrad as OriginalSmoothGrad


class SmoothGrad(Explainability):

    def __init__(self, model):
        self.model = model
        super().__init__('smoothgrad')

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        cuda = torch.cuda.is_available()
        img = img.cuda() if cuda else img
        img = Variable(img, volatile=False, requires_grad=True)
        smooth_grad = OriginalSmoothGrad(image=img, model=self.model, cuda=cuda)
        grad = smooth_grad(idx=target_class)
        grad = self.normalize_grad(grad)
        return grad

    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        img = np.uint8(heatmap * 255)
        return img

    @staticmethod
    def normalize_grad(map):
        saliency = np.max(np.abs(map), axis=1)[0]
        saliency -= saliency.min()
        saliency /= saliency.max()
        return saliency
