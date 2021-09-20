import cv2
import torch
import numpy as np

from explainability.explainability_class import Explainability
from explainability.gradcam.gradcam_original import GradCAM as OriginalGradcam


class Gradcam(Explainability):

    def __init__(self, model):
        self.model = model
        arch = model.__class__.__name__
        target_layers = self.get_target_layers(arch)
        use_cuda = torch.cuda.is_available()
        self.grad_cam = OriginalGradcam(self.model, target_layers, use_cuda)
        super().__init__('gradcam')

    def get_target_layers(self, arch):
        target_layer = {
            'VGG': lambda: self.model.features[-1],
            'ResNet': lambda: self.model.layer4[-1],
            'Inception3': lambda: self.model.Mixed_7c.branch_pool,
            'AlexNet': lambda: self.model.features[-1]
        }
        return target_layer[arch]()

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        grayscale_cam = self.grad_cam(img, target_class)
        return grayscale_cam
    
    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        img = (img - img.min()) / (img.max() - img.min())  # normalized between 0 and 1
        img = img.numpy()[0, :, :, :]  # one image
        img = np.moveaxis(img, 0, 2)  # change color channel position

        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
