import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from .base import normalize


def generate_trajectory(image,
                        reward: float,
                        gamma: float,
                        num_steps: int):
    """Generate a SARS trajectory from a noise to an image"""

    z = tf.random.normal(tf.shape(image), dtype=image.dtype)
    ts = tf.linspace(0.0, 1.0, num_steps)
    trajectory = [z * (1 - t) + image * t for t in ts]

    transitions = []
    for i in range(num_steps-1):
        s = trajectory[i]
        s_hat = trajectory[i+1]
        a = s_hat - s
        transitions.append((s, a, reward * gamma**(num_steps-1-i), s_hat))

    return transitions


def get_sars_dataloaders(batch_size: int=128,
                        reward: float=1.,
                        gamma: float=0.99,
                        num_steps: int=10):
    train_ds = tfds.load('cifar10', as_supervised=True)
    test_ds = tfds.load('cifar10', as_supervised=True)

    def prepare_ds(ds, train=True):
        ds = ds.map(lambda image, _: generate_trajectory(image=image,
                                                     reward=reward,
                                                     gamma=gamma,
                                                     num_steps=num_steps))
        ds = ds.flat_map(lambda transitions: tf.data.Dataset.from_tensor_slices(transitions))

        if train:
            ds = ds.shuffle(buffer_size=5000)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    return prepare_ds(train_ds, train=True), prepare_ds(test_ds, train=False)

