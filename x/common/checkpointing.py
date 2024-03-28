import torch, os
from torch import nn, optim
from pathlib import Path
from x.core.registry import registry
from x.core.datatype import ACheckpointManager


@registry.register_checkpointing(name="CheckpointManager")
class XCheckpointManager(ACheckpointManager):
    """A checkpoint manager saves state dicts of model and optimizer
    as .pth files in a specified directory. This class closely follows
    the API of PyTorch optimizers and learning rate schedulers.

    Note::
        For ``DataParallel`` modules, ``model.module.state_dict()`` is
        saved, instead of ``model.state_dict()``.

    Parameters
    ----------
    model: nn.Module
        Wrapped model, which needs to be checkpointed.
    optimizer: optim.Optimizer
        Wrapped optimizer which needs to be checkpointed.
    checkpoint_dirpath: str
        Path to an empty or non-existent directory to save checkpoints.
    step_size: int, optional (default=1)
        Period of saving checkpoints.
    last_epoch: int, optional (default=-1)
        The index of last epoch.

    Example
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager(model, optimizer, "/tmp/ckpt")
    >>> for epoch in range(20):
    ...     for batch in dataloader:
    ...         do_iteration(batch)
    ...     ckpt_manager.step()
    """

    def __init__(
        self, model, optimizer, checkpoint_dirpath, step_size=1, last_epoch=-1
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("{} is not a Module".format(type(model).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    def step(self, epoch=None):
        """Save checkpoint if step size conditions meet."""
        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if not self.last_epoch % self.step_size:
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_dirpath / f"checkpoint_{self.last_epoch}.pth",
            )

    def model_state_dict(self):
        """Return state dict of model, taking care of DataParallel"""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()

    @staticmethod
    def load_checkpoint(check_point_path):
        print(os.getcwd())
        if os.path.exists(check_point_path):
            raise FileNotFoundError(f"{check_point_path} doesn't exist")
        components = torch.load(check_point_path)
        return components["model"], components["optimizer"]
