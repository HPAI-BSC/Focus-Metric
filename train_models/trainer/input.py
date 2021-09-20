import os
import numpy as np
from torch.utils.data import DataLoader

from consts.consts import Split


class InputPipeline(object):

    def __init__(self, datasets_list, batch_size=1, num_workers=None, seed=None, pin_memory=False):
        if num_workers is None:
            try:
                num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
            except KeyError:
                num_workers = 8

        self.seed = seed
        self.dataloaders = {}
        for ds in datasets_list:
            shuffle = ds.subset == Split.TRAIN
            dl = self.get_dataloader(
                ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            self.dataloaders[ds.subset] = dl

    def _extra_dataloader_kwargs(self, **kwargs):
        if self.seed:
            kwargs['worker_init_fn'] = lambda: np.random.seed(self.seed)
        return kwargs

    def get_dataloader(self, *args, **kwargs):
        kwargs = self._extra_dataloader_kwargs(**kwargs)
        return DataLoader(*args, **kwargs)

    def __getitem__(self, subset):
        try:
            return self.dataloaders[subset]
        except KeyError:
            return None
