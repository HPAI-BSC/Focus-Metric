import time
import torch

from consts.consts import Split
from train_models.utils.pytorch_utils import current_memory_usage, squeeze_generic
from train_models.trainer.saver import load_checkpoint


class Training:
    def __init__(self, input_pipeline, model, loss_function, optimizer, saver, device):
        self.input_pipeline = input_pipeline
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.saver = saver
        self.device = device
        self.initial_epoch = 0

        self.epoch = None

        self._epoch_losses = None
        self._epoch_correct = None
        self._epoch_total = None

    def train_epoch(self, train_loader):
        self.model.train()

        self._epoch_losses = 0
        self._epoch_correct = 0
        self._epoch_total = 0
        t0 = time.time()
        for idx, (batch_images, batch_labels) in enumerate(train_loader):
            # Loading tensors in the used device
            step_images, step_labels = batch_images.to(self.device), batch_labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            step_output, loss = self.forward_to_loss(step_images, step_labels)

            loss.backward()
            self.optimizer.step()
            self.track_metrics(step_labels, loss, step_output)

        train_loss = self._epoch_losses / self._epoch_total
        train_acc = self._epoch_correct / self._epoch_total
        format_args = (self.epoch, train_acc, train_loss, time.time() - t0, current_memory_usage())
        print('EPOCH {} :: train accuracy: {:.4f} - train loss: {:.4f} at {:.1f}s  [{} MB]'.format(*format_args))

    def val_epoch(self, val_loader):
        self.model.eval()

        self._epoch_losses = 0
        self._epoch_correct = 0
        self._epoch_total = 0
        t0 = time.time()
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                # Loading tensors in the used device
                step_images, step_labels = batch_images.to(self.device), batch_labels.to(self.device)
                step_output, loss = self.forward_to_loss(step_images, step_labels)

                self.track_metrics(step_labels, loss, step_output)

        val_loss = self._epoch_losses / self._epoch_total
        val_acc = self._epoch_correct / self._epoch_total
        format_args = (self.epoch, val_acc, val_loss, time.time() - t0, current_memory_usage())
        print('EPOCH {} :: val accuracy: {:.4f} - val loss: {:.4f} at {:.1f}s  [{} MB]'.format(*format_args))

    def forward_to_loss(self, step_images, step_labels):
        step_output = self.model(step_images)
        step_output = squeeze_generic(step_output, axes_to_keep=[0])
        step_labels = squeeze_generic(step_labels, axes_to_keep=[0])
        loss = self.loss_function(step_output, step_labels)
        return step_output, loss

    def track_metrics(self, step_labels, loss, step_output):
        step_total = step_labels.size(0)
        step_loss = loss.item()
        self._epoch_losses += step_loss * step_total
        self._epoch_total += step_total

        step_preds = torch.max(step_output.data, 1)[1]
        step_labels = squeeze_generic(step_labels, axes_to_keep=[0])
        step_correct = (step_preds == step_labels).sum().item()

        self._epoch_correct += step_correct

    def train(self, max_epochs: int):
        for epoch in range(self.initial_epoch, max_epochs):
            self.epoch = epoch
            self.train_epoch(self.input_pipeline[Split.TRAIN])

            if self.saver:
                self.saver.save_checkpoint(self.model, self.optimizer, epoch)

            self.val_epoch(self.input_pipeline[Split.VAL])

    def retrain(self):
        self.model, self.optimizer, last_epoch = load_checkpoint(self.saver.model_path, self.model, self.optimizer)
        self.initial_epoch = last_epoch + 1


class IncpetionTraining(Training):
    def forward_to_loss(self, step_images, step_labels):
        print('is inception True')
        step_output, step_aux_output = self.model(step_images).values()
        step_output = squeeze_generic(step_output, axes_to_keep=[0])
        step_aux_output = squeeze_generic(step_aux_output, axes_to_keep=[0])
        step_labels = squeeze_generic(step_labels, axes_to_keep=[0])
        loss1 = self.loss_function(step_output, step_labels)
        loss2 = self.loss_function(step_aux_output, step_labels)
        loss = loss1 + 0.4 * loss2
        return step_output, loss
