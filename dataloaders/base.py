import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf

import albumentations as A

CIFAR10_MEANS = [0.4914, 0.4822, 0.4465]  # CIFAR-10
CIFAR10_STD = [0.229, 0.224, 0.225]

def get_transforms(train=True):
    if not train:
        return A.Compose([A.Resize(height=32, width=32),
                          A.Normalize(mean=CIFAR10_MEANS, std=CIFAR10_STD)])
    return A.Compose([A.PadIfNeeded(min_height=36, min_width=36, p=1),
                      A.HorizontalFlip(p=0.5),
                      A.RandomCrop(width=32, height=32, p=1),
                      A.ShiftScaleRotate(shift_limit=0.05,
                                         scale_limit=0.1,
                                         rotate_limit=10,
                                         p=0.5),
                      A.OneOf([A.Sharpen(p=1), A.Blur(blur_limit=2, p=1)], p=0.2),
                      A.Normalize(mean=CIFAR10_MEANS, std=CIFAR10_STD)])

def get_base_dataloaders(batch_size=128, num_workers=2):
    train_ds = tfds.load('cifar10', split='train[:90%]', as_supervised=True)
    val_ds = tfds.load('cifar10', split='train[90%:]', as_supervised=True)
    test_ds = tfds.load('cifar10', split='test', as_supervised=True)

    def apply_augmentation(image, label):
        augmented = get_transforms()(image=image.numpy())
        return augmented["image"], label

    def get_dataloader(ds, train=True):
        ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32), tf.cast(lbl, tf.int32)),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda img, lbl: tf.py_function(func=apply_augmentation,
                                                inp=[img, lbl],
                                                Tout=(tf.float32, tf.float32)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if train:
            ds = ds.shuffle(10_000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = get_dataloader(train_ds, train=True)
    val_ds = get_dataloader(val_ds, train=False)
    test_ds = get_dataloader(test_ds, train=False)
    
    return train_ds, val_ds, test_ds