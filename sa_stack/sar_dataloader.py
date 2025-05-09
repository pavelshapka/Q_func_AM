import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jax.numpy as jnp

import math

CIFAR10_MEANS = jnp.array([0.4914, 0.4822, 0.4465])  # CIFAR-10
CIFAR10_STD = jnp.array([0.229, 0.224, 0.225])

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0 

    image = (image - CIFAR10_MEANS) / CIFAR10_STD
    return image, label


def generate_trajectory(image,
                        reward: float,
                        gamma: float,
                        min_num_steps: int,
                        max_num_steps: int,
                        with_reversed_transitions: bool,
                        with_shuffled_transitions: bool):
    """Generate a SARS trajectory from a noise to an image"""

    num_steps = tf.random.uniform((), minval=min_num_steps, maxval=max_num_steps, dtype=tf.int32)
    z = tf.random.normal(tf.shape(image), dtype=image.dtype)

    ts = math.sqrt(2) * tf.range(num_steps, dtype=tf.float32) % 1.0
    ts = tf.sort(ts, direction='ASCENDING') 
    ts = tf.reshape(ts, (num_steps, 1, 1, 1))

    trajectory = z * (1 - ts) + image * ts # [None, 32, 32, 3] * [num_steps, None, None, None]

    s = trajectory[:-1]     # [num_steps-1, 32, 32, 3]
    a = trajectory[1:] - s  # [num_steps-1, 32, 32, 3]
    rewards = reward * (gamma ** tf.reverse(tf.range(num_steps-1, dtype=tf.float32), axis=[0])) # [num_steps-1]
    rewards = tf.reshape(rewards, (-1, 1))

    transitions = tf.concat([s, a], axis=-1) # [num_steps-1, 2 * 32 * 32]

    if with_reversed_transitions:
        trajectory_reverse = tf.reverse(trajectory, axis=[0])

        sr = trajectory_reverse[:-1]
        ar = trajectory_reverse[1:] - sr   # Это "шум, который добавили"
        rewards_r = -reward * (gamma ** tf.reverse(tf.range(num_steps-1, dtype=tf.float32), axis=[0]))
        rewards_r = tf.reshape(rewards_r, (-1, 1))
        transitions_r = tf.concat([sr, ar], axis=-1)

        all_transitions = tf.concat([transitions, transitions_r], axis=0)
        all_rewards = tf.concat([rewards, rewards_r], axis=0)
    else:
        all_transitions = transitions
        all_rewards = rewards

    if with_shuffled_transitions:
        n_total = tf.shape(all_transitions)[0]
        indices = tf.range(n_total)
        shuffled_indices = tf.random.shuffle(indices)
        all_transitions = tf.gather(all_transitions, shuffled_indices)
        all_rewards = tf.gather(all_rewards, shuffled_indices)
    
    return all_transitions, all_rewards


def get_sar_dataloaders(batch_size: int=128,
                        reward: float=1.,
                        gamma: float=0.99,
                        min_num_steps: int=3,
                        max_num_steps: int=10,
                        with_reversed_transitions: bool=False,
                        with_shuffled_transitions: bool=True):
    train_ds = tfds.load('cifar10', split="train", as_supervised=True)
    test_ds = tfds.load('cifar10', split="test", as_supervised=True)

    def prepare_ds(ds, train=True):
        ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda image, _: generate_trajectory(image=image,
                                                         reward=reward,
                                                         gamma=gamma,
                                                         min_num_steps=min_num_steps,
                                                         max_num_steps=max_num_steps,
                                                         with_reversed_transitions=with_reversed_transitions,
                                                         with_shuffled_transitions=with_shuffled_transitions),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda transitions, rewards: tf.data.Dataset.from_tensor_slices((transitions, rewards)))

        if train:
            ds = ds.shuffle(buffer_size=10_000)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    return prepare_ds(train_ds, train=True), prepare_ds(test_ds, train=False)

