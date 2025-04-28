import os
import wandb

from typing import Any
from collections import defaultdict
from pathlib import Path
from coolname import generate_slug
from tqdm import tqdm

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.training import checkpoints
from flax.training import lr_schedule

import optax



CHECKPOINT_PATH = "./checkpoints/td_sarsa"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any

class TrainerModule:

    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 model_hparams : dict[str, Any],
                 batch_size: int,
                 optimizer_name : str,
                 optimizer_hparams : dict[str, Any],
                 exmp_imgs : Any,
                 update_target_every: int = 40,
                 gamma: float = 0.95,
                 ema: float = 0.01,
                 save_every_epoch: int = 5,
                 eval_every_epoch: int = 1,
                 log_every_step: int = 20,
                 mode: str = "opt",
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
        self.batch_size = batch_size
        self.cur_step = 0
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.ema = ema
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        self.save_every_epoch = save_every_epoch
        self.eval_every_epoch = eval_every_epoch
        self.log_every_step = log_every_step
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        self.target_model = self.model_class(**self.model_hparams)
        # Prepare logging
        self.checkpoint_dir = os.path.abspath(os.path.join(CHECKPOINT_PATH, mode, self.model_name))
        self.wandb_logger = None
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)



    def create_functions(self):
        def calculate_loss(params, # Function to calculate the classification loss for a model
                           batch_stats,
                           params_target,
                           batch_stats_target,
                           batch,
                           train: bool,
                           train_rng: jnp.ndarray):
            sarsa_batch, rewards = batch
            states_actions = sarsa_batch[:, :, :, :6]
            next_states_actions = sarsa_batch[:, :, :, 6:]

            outs = self.model.apply({"params": params, "batch_stats": batch_stats},
                                    states_actions,
                                    train=train,
                                    train_rng=train_rng,
                                    mutable=["batch_stats"] if train else False)
            
            q_values, new_model_state = outs if train else (outs, None)

            outs_target = self.target_model.apply({"params": params_target, "batch_stats": batch_stats_target},
                                                  next_states_actions,
                                                  train=False,
                                                  train_rng=None,
                                                  mutable=False)
            
            q_values_target = rewards + self.gamma * outs_target

            loss = optax.l2_loss(q_values, q_values_target).mean()

            return loss, new_model_state
        
        def train_step(state,
                       state_target,
                       batch,
                       train_rng: jnp.ndarray): # Training function
            loss_fn = lambda params: calculate_loss(params=params,
                                                    batch_stats=state.batch_stats,
                                                    params_target=state_target.params,
                                                    batch_stats_target=state_target.batch_stats,
                                                    batch=batch,
                                                    train=True,
                                                    train_rng=train_rng)
            # Get loss, gradients for loss, and other outputs of loss function
            (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            return state, loss
        
        def eval_step(state, state_target, batch): # Eval function. Return the l2 loss for a single batch
            loss, _ = calculate_loss(state.params,
                                     state.batch_stats,
                                     state_target.params,
                                     state_target.batch_stats,
                                     batch,
                                     train=False,
                                     train_rng=None)
            return loss

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params, self.init_batch_stats = variables["params"], variables["batch_stats"]
        self.state = None

        variables_target = self.target_model.init(init_rng, exmp_imgs, train=False)
        self.init_params_target, self.init_batch_stats_target = variables_target["params"], variables_target["batch_stats"]
        self.state_target = None

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
        lr_schedule = optax.piecewise_constant_schedule(init_value=self.optimizer_hparams.pop("lr"),
                                                        boundaries_and_scales={int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                                                                               int(num_steps_per_epoch*num_epochs*0.85): 0.1})
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and "weight_decay" in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop("weight_decay")))
        optimizer = optax.chain(*transf, opt_class(lr_schedule, **self.optimizer_hparams))
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       tx=optimizer)

        self.state_target = TrainState.create(apply_fn=self.target_model.apply,
                                              params=self.init_params_target if self.state_target is None else self.state_target.params,
                                              batch_stats=self.init_batch_stats_target if self.state_target is None else self.state_target.batch_stats,
                                              tx=optax.identity()) # dummy optimizer for target model

    def train_model(self,
                    train_loader,
                    val_loader,
                    rng=jax.random.PRNGKey(42),
                    num_epochs=200,
                    start_from=0):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs - start_from, 1_000)
        if self.wandb_logger is None:
            self.wandb_logger = wandb.init(project="cifar10",
                                           name=self.model_name + "_" + generate_slug(2),
                                           resume="allow")

        for epoch_idx in tqdm(range(start_from, num_epochs+1), initial=start_from, total=num_epochs):
            rng, train_rng = jax.random.split(rng)
            self.train_epoch(train_loader, epoch=epoch_idx, rng=train_rng)
            if (epoch_idx + 1) % self.eval_every_epoch == 0:
                eval_loss = self.eval_model(val_loader)
                self.wandb_logger.log({"val/loss": eval_loss, "epoch": epoch_idx + 1})
            if (epoch_idx + 1) % self.save_every_epoch == 0:
                self.save_model(epoch=epoch_idx + 1)
                

    def update_target_model(self, params, params_target, type: str="soft"):
        if type == "soft":
            return jax.tree_util.tree_map(lambda target, current: (1 - self.ema) * target + self.ema * current, params_target, params)
        elif type == "hard":
            return params
        assert False, f"Unknown target model update type: {type}"

    def train_epoch(self,
                    train_loader,
                    epoch: int,
                    rng: jnp.ndarray): # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in train_loader:
            rng, train_rng = jax.random.split(rng)
            self.state, loss = self.train_step(self.state, self.state_target, tf_to_jax(batch), train_rng)

            metrics["loss"].append(loss)

            self.cur_step += 1
            if self.cur_step % self.update_target_every == 0:
                self.state_target = self.state_target.replace(params=self.update_target_model(self.state.params, self.state_target.params),
                                                              batch_stats=self.update_target_model(self.state.batch_stats, self.state_target.batch_stats))
            if self.cur_step % self.log_every_step == 0:
                log_dict = {"epoch": epoch, "step": self.cur_step}
                for key in metrics:
                    avg_val = jnp.array(metrics[key]).mean()
                    log_dict[f"train/{key}"] = avg_val
                self.wandb_logger.log(log_dict)

                metrics = defaultdict(list)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        total_loss, count = 0, 0
        for batch in data_loader:
            loss = self.eval_step(self.state, self.state_target, tf_to_jax(batch))
            total_loss += loss * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_loss = (total_loss / count).item()
        return eval_loss

    def save_model(self, epoch=0):
        """Save current model"""
        checkpoints.save_checkpoint(ckpt_dir=self.checkpoint_dir,
                                    target={"params": self.state.params,
                                            "batch_stats": self.state.batch_stats,
                                            "epoch": epoch,
                                            "cur_step": self.cur_step,
                                            "wandb_run_id": self.wandb_logger.id,
                                            "wandb_run_step": wandb.run.step},
                                    step=epoch,
                                    overwrite=True)

    def load_model(self):
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.checkpoint_dir, target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       tx=self.state.tx if self.state else optax.sgd(0.1))
        self.cur_step = state_dict.get("cur_step", 0)
        epoch = state_dict.get("epoch", 0)
        wandb_run_id = state_dict.get("wandb_run_id", None)
        wandb_run_step = state_dict.get("wandb_run_step", 0)
        if wandb_run_id is not None:
            self.wandb_logger = wandb.init(project="cifar10", id=wandb_run_id)
        wandb.run.step = wandb_run_step
        print(f"Loaded model from epoch {epoch} with step {self.cur_step}")
        return epoch

    def checkpoint_exists(self) -> bool:
        # Check whether a pretrained model exist for this autoencoder
        return any(item.is_dir() for item in Path(self.checkpoint_dir).iterdir())
    

def tf_to_jax(batch):
    """Конвертирует TF-батч в JAX-совместимый формат."""
    sarsa_batch, rewards = batch[0]._numpy(), batch[1]._numpy()
    return jax.device_put(sarsa_batch), jax.device_put(rewards)