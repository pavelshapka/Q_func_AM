import os

from typing import Any
from collections import defaultdict
from pathlib import Path

import numpy as np

import jax
from flax import linen as nn
from flax.training import train_state
from flax.training import checkpoints
from flax.training import lr_schedule

import optax

from tqdm import tqdm

import wandb


CHECKPOINT_PATH = "./checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any

class TrainerModule:

    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 model_hparams : dict[str, Any],
                 optimizer_name : str,
                 optimizer_hparams : dict[str, Any],
                 exmp_imgs : Any,
                 seed=42):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        # Prepare logging
        self.checkpoint_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.wandb_logger = wandb.init(project="cifar10", name=self.model_name)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params,
                           batch_stats,
                           batch,
                           train):
            imgs, labels = batch
            # Run model. During training, we need to update the BatchNorm statistics.
            outs = self.model.apply({"params": params, "batch_stats": batch_stats},
                                    imgs,
                                    train=train,
                                    mutable=["batch_stats"] if train else False)
            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, new_model_state)
        # Training function
        def train_step(state, batch):
            loss_fn = lambda params: calculate_loss(params,
                                                    state.batch_stats,
                                                    batch,
                                                    train=True)
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, new_model_state = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            return state, loss, acc
        
        def eval_step(state, batch): # Eval function
            # Return the accuracy for a single batch
            _, (acc, _) = calculate_loss(state.params, state.batch_stats, batch, train=False)
            return acc

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params, self.init_batch_stats = variables["params"], variables["batch_stats"]
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop("lr"),
            boundaries_and_scales=
                {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                 int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and "weight_decay" in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop("weight_decay")))
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **self.optimizer_hparams)
        )
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       tx=optimizer)

    def train_model(self,
                    train_loader,
                    val_loader,
                    num_epochs=200,
                    start_from=0):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs - start_from, len(train_loader))

        for epoch_idx in tqdm(range(start_from, num_epochs+1), initial=start_from, total=num_epochs):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 5 == 0:
                eval_acc = self.eval_model(val_loader)
                self.wandb_logger.log({"epoch": epoch_idx, "val/acc": eval_acc}, step=epoch_idx)

    def train_epoch(self, train_loader, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc="Training", leave=False):
            self.state, loss, acc = self.train_step(self.state, tf_to_jax(batch))
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)
        
        log_dict = {"epoch": epoch}
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            log_dict[f"train/{key}"] = avg_val
            
        self.wandb_logger.log(log_dict, step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(self.state, tf_to_jax(batch))
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        """Save current model"""
        abs_ckpt_dir = os.path.abspath(self.checkpoint_dir)
        checkpoints.save_checkpoint(ckpt_dir=abs_ckpt_dir,
                                    target={"params": self.state.params,
                                            "batch_stats": self.state.batch_stats,
                                            "step": step},
                                    step=step,
                                    overwrite=True)

    def load_model(self):
        abs_ckpt_dir = os.path.abspath(self.checkpoint_dir)
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=abs_ckpt_dir, target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       tx=self.state.tx if self.state else optax.sgd(0.1))
        return state_dict.get("step", 0)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        abs_ckpt_dir = os.path.abspath(os.path.join(CHECKPOINT_PATH, self.model_name))
        print(abs_ckpt_dir)
        print(any(item.is_dir() for item in Path(abs_ckpt_dir).iterdir()))

        return any(item.is_dir() for item in Path(abs_ckpt_dir).iterdir())
    

def tf_to_jax(batch):
    """Конвертирует TF-батч в JAX-совместимый формат."""
    images, labels = batch[0]._numpy(), batch[1]._numpy()
    return jax.device_put(images), jax.device_put(labels)