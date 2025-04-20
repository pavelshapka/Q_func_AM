import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from ...classification.default_loader import CIFAR10_MEANS, CIFAR10_STD

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0 

    image = (image - CIFAR10_MEANS) / CIFAR10_STD
    return image, label


def generate_trajectory(image,
                        reward: float,
                        gamma: float,
                        num_steps: int):
    """Generate a SARS trajectory from a noise to an image"""

    z = tf.random.normal(tf.shape(image), dtype=image.dtype)
    ts = tf.linspace(0.0, 1.0, num_steps)
    ts = tf.reshape(ts, (num_steps, 1, 1, 1))

    trajectory = z * (1 - ts) + image * ts # [None, 32, 32, 3] * [num_steps, None, None, None]

    s = trajectory[:-1]     # [num_steps-1, 32, 32, 3]
    s_hat = trajectory[1:]  # [num_steps-1, 32, 32, 3]
    a = s_hat - s           # [num_steps-1, 32, 32, 3]
    rewards = reward * (gamma ** tf.reverse(tf.range(num_steps-1, dtype=tf.float32), axis=[0])) # [num_steps-1]

    transitions = tf.concat([s, a, s_hat], axis=-1) # [num_steps-1, 4 * 32 * 32]
    
    return transitions, rewards


def get_sars_dataloaders(batch_size: int=128,
                         reward: float=1.,
                         gamma: float=0.99,
                         num_steps: int=10):
    train_ds = tfds.load('cifar10', split="train", as_supervised=True)
    test_ds = tfds.load('cifar10', split="test", as_supervised=True)

    def prepare_ds(ds, train=True):
        ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda image, _: generate_trajectory(image=image,
                                                     reward=reward,
                                                     gamma=gamma,
                                                     num_steps=num_steps),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda transitions, rewards: tf.data.Dataset.from_tensor_slices((transitions, rewards)))

        if train:
            ds = ds.shuffle(buffer_size=10_000)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    return prepare_ds(train_ds, train=False), prepare_ds(test_ds, train=False)

