#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-11

from __future__ import print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F


class SmoothGrad(object):

    def __init__(self, image, model, cuda, sigma=0.20, n_samples=10, guided=False):
        self.model = model
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()

        self.sigma = sigma
        self.n_samples = n_samples
        self.image = image

        # Guided Backpropagation
        if guided:
            def func(module, grad_in, grad_out):
                # Pass only positive gradients
                if isinstance(module, nn.ReLU):
                    return (torch.clamp(grad_in[0], min=0.0),)

            for module in self.model.named_modules():
                module[1].register_backward_hook(func)

    def forward(self):
        self.preds = self.model.forward(self.image)
        self.probs = F.softmax(self.preds)[0]
        # self.prob, self.idx = self.probs.data.sort(0, True)
        self.prob = self.probs.data
        return self.prob  # , self.idx

    def encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.probs.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def backward(self, idx):
        # Compute the gradients wrt the specific class
        self.model.zero_grad()
        one_hot = self.encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)

    def __call__(self, idx):
        grads = []
        image = self.image.data.cpu()
        sigma = (image.max() - image.min()) * self.sigma

        for i in range(self.n_samples):
            # Add gaussian noises
            noised_image = image + torch.randn(image.size()) * sigma
            noised_image = noised_image.cuda() if self.cuda else noised_image
            self.image = Variable(
                noised_image, volatile=False, requires_grad=True)
            self.forward()
            self.backward(idx=idx)

            # Sample the gradients on the pixel-space
            grad = self.image.grad.data.cpu().numpy()
            grads.append(grad)

            self.model.zero_grad()

        grad = np.mean(np.array(grads), axis=0)
        return grad


def normalize_grad(map):
    saliency = np.max(np.abs(map), axis=1)[0]
    saliency -= saliency.min()
    saliency /= saliency.max()
    return saliency
