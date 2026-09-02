import numpy as np
from torch.optim import lr_scheduler


class CosineWarmupLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup=0, T_max=10):
        """Warmup + cosine-decay learning rate schedule (step interval).

        Params:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            lr_min (float): Minimum learning rate (floor of the cosine decay).
            lr_max (float): Peak learning rate reached at the end of warmup.
            warmup (int): Number of warmup steps.
            T_max (int): Total number of steps.
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup = warmup
        self.T_max = T_max
        self.cur = 0
        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warmup == 0) and (self.cur == 0):
            lr = self.lr_max
        elif (self.warmup != 0) and (self.cur <= self.warmup):
            lr = self.lr_min + (self.lr_max - self.lr_min) * self.cur / self.warmup
        else:
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (
                np.cos((self.cur - self.warmup) / max(self.T_max - self.warmup, 1) * np.pi) + 1
            )

        self.cur += 1

        return [lr for _ in self.base_lrs]
