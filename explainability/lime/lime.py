import matplotlib.pyplot as plt
import torch
import numpy as np

import torch.nn.functional as F
from torchvision import transforms
from skimage.segmentation import mark_boundaries

from explainability.explainability_class import Explainability
from explainability.lime.utils.unnormalize import UnNormalize
from explainability.lime import lime_image


class Lime(Explainability):

    def __init__(self, model):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()
        super().__init__('lime')

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        img = self.img_unnormalize(img)
        explanation = self.explainer.explain_instance(img,
                                                      self.batch_predict, # classification function
                                                      target_label=target_class,
                                                      hide_color=0,
                                                      num_samples=1000, # number of images that will be sent to classification function
                                                      random_seed=777)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        return mask

    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        img = torch.squeeze(img)
        img = img.permute(1, 2, 0).numpy()
        img = img.astype(np.float64)
        img_boundry = mark_boundaries(img, heatmap)
        return img_boundry

    def batch_predict(self, images):
        self.model.eval()
        preprocess_transform = self.get_preprocess_transform()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.to(device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def get_preprocess_transform(self):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        return transf

    def img_unnormalize(self, img: torch.Tensor) -> np.ndarray:
        img = torch.squeeze(img, 0)
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = unorm(img)
        img = img.permute(1, 2, 0).numpy()
        img *= 255
        img = img.astype(np.uint8)
        return img




