import jax

import tensorflow as tf
import tensorflow_datasets as tfds

def get_image_scaler(): # [0, 1] -> [-1, 1]
    def scaler(x):
        return (x - 0.5)/0.5
    return scaler


def get_image_inverse_scaler(config): # [-1, 1] -> [0, 1]
    def inv_scaler(x):
        return x*0.5 + 0.5
    return inv_scaler

def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2:(h + crop) // 2, 
                  (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(image, tf.uint8)

def get_dataset(config,
                uniform_dequantization=False,
                evaluation=False):
  
    batch_size = config.train.batch_size if not evaluation else config.eval.batch_size

    per_device_batch_size = batch_size // jax.device_count()
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1
    batch_dims = [jax.local_device_count(), per_device_batch_size]

    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'
    
    def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        img = tf.image.convert_image_dtype(d["image"], tf.float32)
        if config.data.random_flip and not evaluation:
            img = tf.image.random_flip_left_right(img)
        if uniform_dequantization:
            img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
        return dict(image=img, label=d.get('label', None))
    
    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.threading.private_threadpool_size = 48
        dataset_options.threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(split=split,
                                            shuffle_files=True,
                                            read_config=read_config)
        else:
            ds = dataset_builder.with_options(dataset_options)

        ds = ds.repeat(count=num_epochs) # None -> бесконечное количество эпох
        if not evaluation:
            ds = ds.shuffle(shuffle_buffer_size) # перемешивание
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for batch_size in reversed(batch_dims):
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size) # загружает данные в фоновом режиме

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds, dataset_builder
