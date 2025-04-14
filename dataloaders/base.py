import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf

DATA_MEANS = jnp.array([0.4914, 0.4822, 0.4465])  # CIFAR-10
DATA_STD = jnp.array([0.2470, 0.2435, 0.2616])


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0 

    image = (image - DATA_MEANS) / DATA_STD
    return image, label


def get_base_dataloaders(batch_size=128, num_workers=2):
    train_ds = tfds.load('cifar10', split='train[:90%]', as_supervised=True)
    val_ds = tfds.load('cifar10', split='train[90%:]', as_supervised=True)
    test_ds = tfds.load('cifar10', split='test', as_supervised=True)

    def get_dataloader(ds, train=True):
        if train:
            ds = ds.shuffle(10000)
        ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = get_dataloader(train_ds, train=True)
    val_ds = get_dataloader(val_ds, train=False)
    test_ds = get_dataloader(test_ds, train=False)
    
    return train_ds, val_ds, test_ds