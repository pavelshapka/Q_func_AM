import os
import wandb

from typing import Any
from collections import defaultdict
from pathlib import Path
from coolname import generate_slug

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.training import checkpoints
from flax.core import freeze, unfreeze
import time

import math
from . import dataset

import optax



CHECKPOINT_PATH = "./checkpoints/td_sarsa"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

class TrainState(train_state.TrainState):
    batch_stats: Any

class TrainerModuleSingle:

    def __init__(self,
                 config,
                 model_class: nn.Module,
                 version: int = 1):
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
        self.seed = config.seed
        self.wandb_track = config.wandb_track
        self.num_devices = jax.local_device_count()

        self.model_name = config.model.name
        self.model_class = model_class
        activations = {"swish": nn.silu,
                       "relu": nn.relu}
        model_hparams = {"activation": activations[config.model.activation]}
        self.model = self.model_class(**model_hparams)
        self.target_model = self.model_class(**model_hparams)

        self.n_steps = config.train.n_steps
        self.batch_size = config.train.batch_size
        self.update_target_every = config.train.update_target_every
        self.log_every = config.train.log_every
        self.save_every = config.train.save_every

        self.gamma = config.data.gamma
        self.ema = config.train.ema

        self.create_functions(config)    # Create jitted training and eval functions
        self.init_model(config.model.optimizer, {"lr": config.train.lr, "weight_decay": config.model.optimizer_weight_decay}) # Initialize model

        # Prepare logging
        self.checkpoint_dir = os.path.abspath(os.path.join(CHECKPOINT_PATH, f"{self.model_name}_{str(version)}"))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if self.checkpoint_exists():
            self.load_model()
            return
        
        self.initial_step = 0
        if self.wandb_track:
            self.wandb_logger = wandb.init(project="cifar10",
                                           name=self.model_name + "_" + generate_slug(2),
                                           resume="allow")


    def create_functions(self, config):
        def calculate_loss(variables: dict,
                           variables_target: dict,
                           batch: tuple[jnp.ndarray, jnp.ndarray],
                           train: bool,
                           rng_key: jax.Array):
            sarsa_batch, rewards = batch
            states_actions = sarsa_batch[:, :, :, :6]
            next_states_actions = sarsa_batch[:, :, :, 6:]

            outs = self.model.apply(variables,
                                    states_actions,
                                    train=train,
                                    train_rng=rng_key,
                                    mutable=["batch_stats"] if train else False)
            
            q_values, new_model_state = outs if train else (outs, None)

            outs_target = self.target_model.apply(variables_target,
                                                  next_states_actions,
                                                  train=False,
                                                  train_rng=None,
                                                  mutable=False)
            
            q_values_target = rewards + self.gamma * outs_target

            loss = optax.l2_loss(q_values, q_values_target).mean()
            return loss, new_model_state
        
        def train_step(state: TrainState,
                           state_target: TrainState,
                           batch: tuple[jnp.ndarray, jnp.ndarray],
                           rng_key: jax.Array):
            
            loss_fn = lambda params: calculate_loss(variables={"params": params, "batch_stats":state.batch_stats},
                                                    variables_target={"params": state_target.params, "batch_stats": state_target.batch_stats},
                                                    batch=batch,
                                                    train=True,
                                                    rng_key=rng_key)

            (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
            
            return state, loss

        def sync_batch_stats(state):
            """Synchronize batch statistics across devices."""
            if state.batch_stats is not None:
                batch_stats = jax.lax.pmean(state.batch_stats, axis_name='devices')
                return state.replace(batch_stats=batch_stats)
            return state

        self.train_step = jax.jit(train_step)

    def init_model(self,
                   opt_name: str,
                   opt_hparams: dict):
        init_rng = jax.random.PRNGKey(self.seed)

        # per_device_batch_size = self.batch_size // self.num_devices
        init_batch = jnp.zeros((self.batch_size, 32, 32, 6), dtype=jnp.float32)
        
        print(f"Initializing model with batch shape: {init_batch.shape}")
        
        variables = self.model.init(init_rng, init_batch, train=True)
        variables_target = self.target_model.init(init_rng, init_batch, train=False)

        opt_classes = {"adam": optax.adam,
                       "adamw": optax.adamw}
        opt_class = opt_classes[opt_name.lower()]
        lr = opt_hparams.pop("lr")
        optimizer = optax.chain(optax.clip(1.0), opt_class(lr, **opt_hparams))
        
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=variables["params"],
                                       batch_stats=variables["batch_stats"],
                                       tx=optimizer)

        self.state_target = TrainState.create(apply_fn=self.target_model.apply,
                                              params=variables_target["params"],
                                              batch_stats=variables_target["batch_stats"],
                                              tx=optax.identity()) # dummy optimizer for target model

    def train_model(self,
                    train_ds,
                    rng_key: jax.Array | None = None):
        rng_key = rng_key or jax.random.PRNGKey(self.seed)
        metrics = defaultdict(list)
        batch_time = time.time()

        train_iter = iter(train_ds)
        scaler = dataset.get_image_scaler()

        for step in range(self.initial_step, self.n_steps):
            rng_key, train_rng_key = jax.random.split(rng_key, num=2)
            # train_rng_keys = jax.random.split(train_rng_key, self.num_devices)
            
            batch = jax.tree.map(lambda x: scaler(x.numpy()), next(train_iter))
            self.state, loss = self.train_step(state=self.state,
                                                    state_target=self.state_target,
                                                    batch=batch,
                                                    rng_key=train_rng_key)

            loss = loss
            
            if not self.wandb_track:
                continue

            metrics["loss"].append(loss)

            if step % self.log_every == 0 and step != self.initial_step:
                log_dict = {"step": step, "batch_time": time.time() - batch_time}
                for key in metrics:
                    avg_val = jnp.array(metrics[key]).mean()
                    log_dict[f"train/{key}"] = avg_val
                self.wandb_logger.log(log_dict)

                metrics = defaultdict(list)
                batch_time = time.time()

            if step % self.update_target_every == 0 and step != self.initial_step:
                self.state_target = self.update_target(mode="soft")

            if step % self.save_every == 0 and step != self.initial_step:
                self.save_model(step=step)        

    def update_target(self, mode: str="soft"):
        update_fns = {"soft": lambda current, target: (1 - self.ema) * current + self.ema * target, # 0.01 * cur + 0.99 * targ
                      "hard": lambda current, target: current}
        update_fn = update_fns[mode]

        # state = self._get_first_device_state(self.state)
        # state_target = self._get_first_device_state(self.state_target)
        
        params_new = jax.tree_util.tree_map(update_fn, self.state.params, self.state_target.params)
        batch_stats_new = jax.tree_util.tree_map(update_fn, self.state.batch_stats, self.state_target.batch_stats)
        
        new_state_target = self.state_target.replace(params=params_new, batch_stats=batch_stats_new)

        return new_state_target

    def save_model(self, step: int=0):
        """Save current model"""
        
        checkpoints.save_checkpoint(ckpt_dir=self.checkpoint_dir,
                                    target={"params": self.state.params,
                                            "batch_stats": self.state.batch_stats,
                                            "step": step,
                                            "wandb_run_id": self.wandb_logger.id if self.wandb_track else None,
                                            "wandb_run_step": wandb.run.step if self.wandb_track else 0},
                                    step=step,
                                    overwrite=False)

    def load_model(self) -> None:
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.checkpoint_dir, target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict["params"],
                                       batch_stats=state_dict["batch_stats"],
                                       tx=self.state.tx)
        
        self.state_target = TrainState.create(apply_fn=self.target_model.apply,
                                              params=state_dict["params"],
                                              batch_stats=state_dict["batch_stats"],
                                              tx=optax.identity()) # dummy optimizer for target model
            
        self.initial_step = state_dict.get("step", 0)
        wandb_run_id = state_dict.get("wandb_run_id", None)
        if wandb_run_id is not None and self.wandb_track:
            self.wandb_logger = wandb.init(project="cifar10", id=wandb_run_id)

        print(f"Loaded model from step {self.initial_step}")

    def checkpoint_exists(self) -> bool:
        return any(item.is_dir() for item in Path(self.checkpoint_dir).iterdir())
