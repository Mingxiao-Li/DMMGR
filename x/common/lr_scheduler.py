from torch.optim import Optimizer
from x.core.registry import registry
from x.core.datatype import ALRScheduler


@registry.register_lrscheduler(name="LRScheduler")
class XLRScheduler(ALRScheduler):
    def __init__(
        self,
        lr_base: float,
        optimizer: Optimizer,
        method: str,
        step: int = None,
        decay_rate: float = None,
        step2: int = None,
        total_step: int = None,
        min_lr: float = 1e-5,
    ) -> None:
        r"""  A learning rate schedule class which can realzie learning rate decay and learning rate warmup
         !!! one step means one epoch
        :param lr_base: base learning rate
        :param optimizer: optimizer
        :param step / step2 / total_step :
                               for "decay" method, only steo is needed
                               for "warmup" method, all step, step2, total_step are needed
                               step, step2, total_step
                               1. warmup wothout decay ==> step2=total_step
                                  learnring rate plot:
                                                      _____________
                                                    /
                                                  /  step
                               2. warmup with decay  ==> step < step2 < toal_step
                                  learning rate plot:
                                                      ______________
                                                    /               \
                                                  /  step       step2\ total_step
                               3. warmup  ==> step==step2 < total_step
                                  learning raet plot:
                                                         /\
                                                       /   \
                                                     / step \ total_step
        :param method: "decay" or "warmup"
        :param min_lr: minimal learning rate
        """
        assert method in [
            "decay",
            "warmup",
        ], "Learning rate schdedule method can only be 'decay' or warmup"

        self.optimizer = optimizer
        self._steps = 0
        self._decay_rate = decay_rate
        self._lr_base = lr_base
        self._min_lr = min_lr
        self._step = step
        self._step2 = step2
        self._total_step = total_step
        self._method = method

    def step(self):
        self._steps += 1
        lr = self.update_lr()
        for p in self.optimizer.param_groups:
            p["lr"] = lr

    def decay(self, step: int) -> float:
        r"""Two decay methods.
        when decay_rate is set, decay by teh decay rate
        otherwise decay_rate if calculated using steps
        """
        assert step is not None or self._decay_rate is not None
        if self._decay_rate is not None:
            lr = self._lr_base * (self._decay_rate ** step)
            lr = lr if lr > self._min_lr else self._min_lr
            return lr
        if step <= self._step:
            lr_factor = float(max(1, self._step - step)) / float(max(1, self._step))
            lr = (
                self._lr_base * lr_factor
                if self._lr_base * lr_factor > self._min_lr
                else self._min_lr
            )
        else:
            lr_factor = 1.0 / float(max(1, self._step))
            lr = (
                self._lr_base * lr_factor
                if self._lr_base * lr_factor > self._min_lr
                else self._min_lr
            )
        return lr

    def warmup(self, step: int) -> float:
        assert (
            self._step2 is not None and self._total_step is not None
        ), "For warmup method, step2 and total_step should not be None !!"
        if step <= self._step:
            lr_factor = float(step) / float(max(1, self._step))
            lr = (
                self._lr_base * lr_factor
                if self._lr_base * lr_factor > self._min_lr
                else self._min_lr
            )
        elif step <= self._step2:
            lr = self._lr_base
        elif self._step2 < step <= self._total_step:
            lr_factor = max(1, float(self._total_step - step)) / float(
                max(1.0, self._total_step - self._step2)
            )
            lr = (
                self._lr_base * lr_factor
                if self._lr_base * lr_factor > self._min_lr
                else self._min_lr
            )
        else:
            lr = self._min_lr
        return lr

    def update_lr(self, step=None) -> float:
        if step is None:
            step = self._steps
        if self._method == "decay":
            return self.decay(step)
        if self._method == "warmup":
            return self.warmup(step)
