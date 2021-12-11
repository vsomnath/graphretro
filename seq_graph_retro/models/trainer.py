import torch
import torch.nn as nn
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import sys
import wandb
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from seq_graph_retro.utils.torch import EncOptimizer


class Trainer:
    """Trainer class for training models and storing summaries."""

    def __init__(self,
                 model: nn.Module,
                 ckpt_dir: str = "./checkpoints",
                 log_dir: str = "./logs",
                 eval_metric: str = 'accuracy',
                 add_grad_noise: bool = False,
                 print_every: int = 100,
                 eval_every: int = None,
                 save_every: int = None,
                 **kwargs):
        """
        Parameters
        ----------
        model: nn.Module,
            Model to train and evaluate
        ckpt_dir: str, default ./checkpoints
            Directory to save checkpoints to.
        lr: float, default 0.001
            Learning rate, used only when optimizer is None
        optimizer: torch.optim.Optimizer, default None
            Optimizer used
        scheduler: torch.optim.lr_scheduler, default None,
            Learning rate scheduler used.
        print_every: int, default 100
            Print stats every print_every iterations
        eval_every: int, default None,
            Frequency of evaluation during training. If None, evaluation done
            only every epoch
        """
        self.model = model
        if add_grad_noise:
            for param in self.model.parameters():
                param.register_hook(self.grad_with_noise)
        self.print_every = print_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.eval_metric = eval_metric
        if 'acc' in eval_metric:
            self.best_metric = 0.0
        elif 'loss' in eval_metric:
            self.best_metric = np.inf
        self.global_step = 0
        self.epoch_start = 0

    def build_optimizer(self, learning_rate: float, finetune_encoder: bool) -> torch.optim.Optimizer:
        def encoder_param_cond(key: str) -> bool:
            return 'encoder' in key

        if finetune_encoder:
            net_params = [v for key, v in self.model.named_parameters()
                          if not encoder_param_cond(key)]
            enc_params = [v for key, v in self.model.named_parameters()
                          if encoder_param_cond(key)]
        else:
            net_params = [v for key, v in self.model.named_parameters()]
            enc_params = []

        net_optimizer = torch.optim.Adam(net_params, lr=learning_rate)
        if not enc_params:
            enc_optimizer = None
        else:
            enc_optimizer = torch.optim.Adam(enc_params, lr=1e-4)
        self.optimizer = EncOptimizer(optimizer=net_optimizer, enc_opt=enc_optimizer)

    def build_scheduler(self, type: str, anneal_rate: float, patience: Optional[int] = None,
                        thresh: Optional[float] = None) -> torch.optim.lr_scheduler:
        if type == 'exp':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, anneal_rate)
        elif type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                       patience=patience,
                                                       factor=anneal_rate,
                                                       threshold=thresh,
                                                       threshold_mode='abs')
        else:
            self.scheduler = None

    def _save_epoch(self, epoch: int) -> None:
        """Saves checkpoint after epoch.

        Parameters
        ----------
        epoch: int,
            Epoch number
        """
        name = f"epoch_{epoch}.pt"
        self._save_checkpoint(name=name)

    def _save_checkpoint(self, name: str = None) -> None:
        """Saves checkpoint.

        Parameters
        ----------
        name: str, default None
            Name of the checkpoint.
        """
        save_dict = {'state': self.model.state_dict()}
        if hasattr(self.model, 'get_saveables'):
            save_dict['saveables'] = self.model.get_saveables()

        if name is None:
            name = f"best_model.pt"
        save_file = os.path.join(wandb.run.dir, name)
        torch.save(save_dict, save_file)

    def log_metrics(self, metrics, mode='train'):
        metric_dict = {}
        metric_dict['iteration'] = self.global_step
        for metric in metrics:
            if metrics[metric] is not None:
                metric_dict[f"{mode}_{metric}"] = metrics[metric]

        wandb.log(metric_dict)

    def train_epochs(self, train_data: DataLoader, eval_data: DataLoader, epochs: int = 10, **kwargs) -> None:
        """Train model for given number of epochs.

        Parameters
        ----------
        data: MolGraphDataset
            Dataset to generate batches from.
        batch_size: int, default 16
            Batch size used for training
        epochs: int, default 10
            Number of epochs used for training
        """
        for epoch in range(epochs):
            print(f"--------- Starting Epoch: {self.epoch_start + epoch+1} ----------------")
            print()
            sys.stdout.flush()

            epoch_metrics = self._train_epoch(train_data, eval_data, **kwargs)
            for metric, val in epoch_metrics.items():
                epoch_metrics[metric] = np.round(np.mean(val), 4)
            metrics = self._evaluate(eval_data, **kwargs)

            # Check if learning rate needs to be changed
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ExponentialLR):
                self.scheduler.step()

            elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if 'acc' in self.eval_metric:
                    assert self.scheduler.mode == 'max'
                    eval_acc = metrics.get(self.eval_metric, None)
                    if eval_acc is not None:
                        self.scheduler.step(eval_acc)

                elif 'loss' in self.eval_metric:
                    assert self.scheduler.mode == 'min'
                    eval_loss = metrics.get(self.eval_metric, None)
                    if eval_loss is not None:
                        self.scheduler.step(eval_loss)

            else:
                pass

            # Updates best metric for saving checkpoints
            if 'acc' in self.eval_metric:
                eval_acc = metrics.get(self.eval_metric, None)
                if eval_acc is not None and eval_acc > self.best_metric:
                    self.best_metric = eval_acc
                    self._save_checkpoint()

            elif 'loss' in self.eval_metric:
                eval_loss = metrics.get(self.eval_metric, None)
                if eval_loss is not None and eval_loss < self.best_metric:
                    self.best_metric = eval_loss
                    self._save_checkpoint()

            else:
                pass

            print(f"-------- Completed Epoch: {epoch+1} Global Step: {self.global_step} ----------------")
            print(f"Train Metrics: {epoch_metrics}")
            print(f"Eval Metrics: {metrics}")
            print("-----------------------------------------------------")
            print()

            sys.stdout.flush()

    def _train_epoch(self, train_data: DataLoader, eval_data: DataLoader = None, **kwargs) -> List[np.ndarray]:
        """Train a single epoch of data.

        Parameters
        ----------
        data: MolGraphDataset
            Dataset to generate batches from, mode == eval
        batch_size: int, default 16
            batch size used for training
        """
        epoch_losses = []
        epoch_metrics = {}

        for idx, inputs in enumerate(train_data):
            self.global_step += 1

            if idx % self.print_every == 0:
                print(f"After {idx+1} steps, Global Step: {self.global_step}")
                sys.stdout.flush()

            n_elem = len(inputs[-1])
            step_metrics = self._train_step(inputs=inputs, step_count=idx, **kwargs)

            for metric, metric_val in step_metrics.items():
                if metric not in epoch_metrics:
                    epoch_metrics[metric] = [metric_val] * n_elem
                else:
                    epoch_metrics[metric].extend([metric_val] * n_elem)
            epoch_losses.extend([step_metrics["loss"]] * n_elem)

            if idx % self.print_every == 0:
                metrics = epoch_metrics.copy()
                for metric, metric_vals in metrics.items():
                    metrics[metric] = np.round(np.mean(metric_vals), 4)

                print(f"Train Metrics so far: {metrics}")
                print()
                self.log_metrics(metrics, mode='train')
                sys.stdout.flush()

            if self.eval_every is not None:
                if idx % self.eval_every == 0 and idx:
                    eval_metrics = self._evaluate(eval_data, **kwargs)
                    print(f"Evaluating after {idx+1} steps, Global Step: {self.global_step}")
                    print(f"Eval Metrics: {eval_metrics}")
                    sys.stdout.flush()

                    if 'acc' in self.eval_metric:
                        eval_acc = eval_metrics.get(self.eval_metric, None)
                        if eval_acc is not None and eval_acc > self.best_metric:
                            self.best_metric = eval_acc
                            print(f"Global Step: {self.global_step}. Best eval accuracy so far. Saving model.")
                            sys.stdout.flush()
                            self._save_checkpoint()

                    elif 'loss' in self.eval_metric:
                        eval_loss = eval_metrics.get(self.eval_metric, None)
                        if eval_loss is not None and eval_loss < self.best_metric:
                            self.best_metric = eval_loss
                            print(f"Global Step: {self.global_step}. Best eval loss so far. Saving model.")
                            sys.stdout.flush()
                            self._save_checkpoint()

                    else:
                        pass

                    print()
                    sys.stdout.flush()

            if self.save_every is not None:
                if idx % self.save_every == 0 and idx:
                    print(f"Saving model after global step {self.global_step}")
                    print()
                    sys.stdout.flush()
                    self._save_checkpoint()

        return epoch_metrics

    def _evaluate(self, eval_data: DataLoader, **kwargs) -> Dict[str, float]:
        """Computes metrics on eval dataset.

        Parameters
        ----------
        data: MolGraphDataset
            Dataset to generate batches from, mode == eval
        batch_size: int, default 1
            batch size used for evaluation
        """
        eval_metrics = {}
        self.model.eval()

        if eval_data is None:
            self.model.train()
            return eval_metrics

        for idx, inputs in enumerate(eval_data):
            metrics = self._eval_step(inputs, **kwargs)
            if not len(eval_metrics):
                eval_metrics = {key: [] for key in metrics}

            for metric in metrics:
                eval_metrics[metric].append(metrics[metric])

        for metric in eval_metrics:
            if None not in eval_metrics[metric]:
                eval_metrics[metric] = np.round(np.mean(eval_metrics[metric]), 4)
            else:
                eval_metrics[metric] = None

        self.log_metrics(eval_metrics, mode='eval')
        self.model.train()
        return eval_metrics

    def _eval_step(self, inputs: Tuple[Tuple[torch.Tensor, ...], ...], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Tuple[np.ndarray, np.ndarray]]:
        """Runs an eval step.

        Parameters
        ----------
        inputs: tuple of tuples of torch.Tensors
            Inputs to the WLNDisconnect forward pass
        """
        with torch.no_grad():
            eval_loss, eval_metrics = self.model.eval_step(*inputs, **kwargs)
            if eval_loss is not None:
                assert torch.isfinite(eval_loss).all()
            for metric in eval_metrics:

                if eval_metrics[metric] is not None:
                    eval_metrics[metric] = np.round(eval_metrics[metric], 4)
        return eval_metrics

    def grad_with_noise(self, grad):
        std = np.sqrt(1.0 / (1 + self.global_step) ** 0.55)
        noise = std * torch.randn(tuple(grad.shape), device=grad.device)
        return grad + noise

    def _train_step(self, inputs: Tuple[Tuple[torch.Tensor, ...], ...],
                    step_count: int, **kwargs) -> Dict[str, float]:
        """Runs a train step.

        Parameters
        ----------
        inputs: tuple of tuples of torch.Tensors
            Inputs to the WLNDisconnect forward pass
        optimizer: torch.optim.Optimizer:
            optimizer used for gradient computation
        """
        total_loss, metrics = self.model.train_step(*inputs)
        assert torch.isfinite(total_loss).all()

        accum_every = kwargs.get('accum_every', None)
        if accum_every is not None:
            apply_grad = (step_count % accum_every) == 0
            total_loss /= accum_every
            total_loss.backward()

            if apply_grad:
                if "clip_norm" in kwargs:
                    nn.utils.clip_grad_norm_(self.model.parameters(), kwargs["clip_norm"])

                self.optimizer.step()
                self.optimizer.zero_grad()

                if step_count % self.print_every == 0:
                    if torch.cuda.is_available():
                        alloc_memory = torch.cuda.memory_allocated() / 1024.0 / 1024.0
                        cached_memory = torch.cuda.memory_reserved() / 1024.0 / 1024.0
                        print(f"Memory: Allocated: {alloc_memory:.3f} MB, Cache: {cached_memory:.3f} MB")
                        sys.stdout.flush()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            self.optimizer.zero_grad()
            total_loss.backward()

            if "clip_norm" in kwargs:
                nn.utils.clip_grad_norm_(self.model.parameters(), kwargs["clip_norm"])

            self.optimizer.step()

            if step_count % self.print_every == 0:
                if torch.cuda.is_available():
                    alloc_memory = torch.cuda.memory_allocated() / 1024.0 / 1024.0
                    cached_memory = torch.cuda.memory_reserved() / 1024.0 / 1024.0
                    print(f"Memory: Allocated: {alloc_memory:.3f} MB, Cache: {cached_memory:.3f} MB")
                    sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for metric in metrics:
            metrics[metric] = np.round(metrics[metric], 4)

        return metrics
