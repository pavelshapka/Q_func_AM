import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf

# Константы для нормализации
DATA_MEANS = jnp.array([0.4914, 0.4822, 0.4465])  # Пример для CIFAR-10
DATA_STD = jnp.array([0.2470, 0.2435, 0.2616])


def random_crop(image, crop_rng):
    crop = jax.random.uniform(crop_rng, shape=(), minval=0.8, maxval=1.0)
    h, w = image.shape[:2]
    new_h, new_w = int(h * crop), int(w * crop)
    y = jax.random.randint(crop_rng, shape=(), minval=0, maxval=h-new_h)
    x = jax.random.randint(crop_rng, shape=(), minval=0, maxval=w-new_w)
    image = jax.lax.dynamic_slice(image, (y, x, 0), (new_h, new_w, 3))
    image = jax.image.resize(image, (h, w, 3), method='bilinear')
    return image


def normalize(image, label):
    image = jnp.array(image, dtype=jnp.float32) / 255

    image = (image - DATA_MEANS) / DATA_STD
    return image


def get_dataset(batch_size=128, num_workers=2):
    train_ds = tfds.load('cifar10', split='train[:90%]', as_supervised=True)
    val_ds = tfds.load('cifar10', split='train[90%:]', as_supervised=True)
    test_ds = tfds.load('cifar10', split='test', as_supervised=True)
    
    train_ds = train_ds.shuffle(10000)\
                       .map(normalize)\
                       .batch(batch_size)\
                       .prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds