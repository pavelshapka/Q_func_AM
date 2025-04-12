from typing import Callable

import functools

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.training import train_state, checkpoints
main_rng = random.PRNGKey(42)

from torchvision.datasets import CIFAR10

from preprocess import get_dataset


class ConvNxN(nn.Module):
    N: int
    out_channels: int
    stride: int = 1
    padding: str = "SAME"
    activation: Callable = nn.relu
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Conv(features=self.out_channels,
                    kernel_size=(self.N, self.N),
                    kernel_init=nn.initializers.kaiming_normal(),
                    strides=(self.stride, self.stride),
                    padding=self.padding,
                    use_bias=self.use_bias)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.activation(x)
        return x
    

class InceptionBlock(nn.Module):
    out_channels: dict[str, int]
    reduced_channels: dict[str, int]
    activation: Callable = nn.relu
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        conv_1x1 = functools.partial(ConvNxN, N=1, activation=self.activation, padding=self.padding)
        conv_3x3 = functools.partial(ConvNxN, N=3, activation=self.activation, padding=self.padding)
        conv_5x5 = functools.partial(ConvNxN, N=5, activation=self.activation, padding=self.padding)
                                     
        x_1x1 = conv_1x1(out_channels=self.out_channels["conv1x1"],
                         stride=2,
                         use_bias=False)(x, train)
        
        x_3x3 = conv_1x1(out_channels=self.reduced_channels["conv3x3"],
                         stride=1,
                         use_bias=False)(x, train)
        x_3x3 = conv_3x3(out_channels=self.out_channels["conv3x3"],
                         stride=2,
                         use_bias=False)(x_3x3, train)
        
        x_5x5 = conv_1x1(out_channels=self.reduced_channels["conv5x5"],
                         stride=1,
                         use_bias=False)(x, train)
        x_5x5 = conv_5x5(out_channels=self.out_channels["conv5x5"],
                         stride=2,
                         use_bias=False)(x_5x5, train)
        
        x_mp = nn.max_pool(inputs=x,
                           window_shape=(3, 3),
                           strides=(2, 2),
                           padding="SAME")
        x_mp = conv_1x1(out_channels=self.out_channels["max_pool"],
                        stride=1,
                        use_bias=False)(x_mp, train)
        
        x = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_mp], axis=-1)
        return x
        

class InceptionNet(nn.Module):
    num_classes: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = ConvNxN(N=3,
                    out_channels=64,
                    stride=1,
                    padding="SAME",
                    activation=self.activation,
                    use_bias=False)(x, train)
        
        x = InceptionBlock(out_channels={"conv1x1": 16, "conv3x3": 32, "conv5x5": 8, "max_pool": 8},
                           reduced_channels={"conv3x3": 32, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 24, "conv3x3": 48, "conv5x5": 12, "max_pool": 12},
                           reduced_channels={"conv3x3": 32, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = nn.max_pool(inputs=x,
                        window_shape=(3, 3),
                        strides=(2, 2),
                        padding="SAME")
        
        x = InceptionBlock(out_channels={"conv1x1": 24, "conv3x3": 48, "conv5x5": 12, "max_pool": 12},
                           reduced_channels={"conv3x3": 32, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 16, "conv3x3": 48, "conv5x5": 16, "max_pool": 16},
                           reduced_channels={"conv3x3": 32, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 16, "conv3x3": 48, "conv5x5": 16, "max_pool": 16},
                           reduced_channels={"conv3x3": 32, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 32, "conv3x3": 48, "conv5x5": 24, "max_pool": 24},
                           reduced_channels={"conv3x3": 32, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = nn.max_pool(inputs=x,
                        window_shape=(3, 3),
                        strides=(2, 2),
                        padding="SAME")
        
        x = InceptionBlock(out_channels={"conv1x1": 32, "conv3x3": 64, "conv5x5": 16, "max_pool": 16},
                           reduced_channels={"conv3x3": 48, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 32, "conv3x3": 64, "conv5x5": 16, "max_pool": 16},
                           reduced_channels={"conv3x3": 48, "conv5x5": 16},
                           activation=self.activation)(x, train)
        
        x = x.mean(axis=(1, 2))
        x = nn.Dense(features=self.num_classes,
                     kernel_init=nn.initializers.kaiming_normal(),
                     use_bias=False)(x)
        return x
