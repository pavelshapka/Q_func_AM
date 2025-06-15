import jax
import math

import tensorflow as tf
import tensorflow_datasets as tfds

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
    num_devices = jax.local_device_count()
    per_device_batch_size = batch_size // jax.device_count()
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1
    batch_dims = [num_devices, per_device_batch_size] if config.multi_device else [batch_size]
    
    print(f"Batch dimensions: {batch_dims}")

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

        img = (img-0.5)/0.5
        return img, d.get('label', None)
    
    def generate_sarsa_opt_trajectory(image,
                                      reward: float,
                                      gamma: float,
                                      min_num_steps: int,
                                      max_num_steps: int,
                                      with_reversed_actions: bool = False,
                                      with_random_actions: bool = False):
        """Generate a SARS trajectory from a noise to an image"""

        num_steps = tf.random.uniform((), minval=min_num_steps, maxval=max_num_steps, dtype=tf.int32)
        z = tf.random.normal(tf.shape(image), dtype=image.dtype)

        ts = math.sqrt(2) * tf.range(num_steps, dtype=tf.float32) % 1.0 # Имитация равномерного распределения
        ts = tf.sort(ts, direction='ASCENDING')
        ts = tf.reshape(ts, (num_steps, 1, 1, 1))

        trajectory = z * (1 - ts) + image * ts # [None, 32, 32, 3] * [num_steps, None, None, None]

        s = trajectory[:-1]                # [num_steps-1, 32, 32, 3]
        s_next = trajectory[1:]            # [num_steps-1, 32, 32, 3]
        a = s_next - s                     # [num_steps-1, 32, 32, 3]
        a_next = image[None, ...] - s_next # [num_steps-1, 32, 32, 3]

        rewards = reward * (gamma ** tf.range(num_steps-1, 0, -1, dtype=tf.float32)) # [num_steps-1]

        transitions = tf.concat([s, a, s_next, a_next], axis=-1) # [num_steps-1, 32, 32, 3*4=12]

        if with_reversed_actions:
            s_rev = trajectory[1:]
            s_next_rev = trajectory[:-1]
            a_rev = s_next_rev - s_rev
            a_next_rev = image[None, ...] - s_next_rev
            transitions_reversed = tf.concat([s_rev, a_rev, s_next_rev, a_next_rev], axis=-1)

            rewards_reversed = -reward * (gamma ** tf.range(num_steps-1, 0, -1, dtype=tf.float32)) # [num_steps-1]
            rewards = tf.concat([rewards, rewards_reversed], axis=0)
            transitions = tf.concat([transitions, transitions_reversed], axis=0)

        if with_random_actions:
            s_rand = trajectory
            a_rand = tf.random.normal(tf.shape(trajectory), dtype=image.dtype) / tf.cast(num_steps, dtype=image.dtype)
            s_next_rand = s_rand + a_rand
            a_next_rand = image[None, ...] - s_next_rand
            transitions_random = tf.concat([s_rand, a_rand, s_next_rand, a_next_rand], axis=-1)

            rewards_random = tf.zeros(num_steps, dtype=tf.float32) # [num_steps]
            rewards = tf.concat([rewards, rewards_random], axis=0)
            transitions = tf.concat([transitions, transitions_random], axis=0)


        rewards = tf.reshape(rewards, (-1, 1))
        return transitions, rewards
    
    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.threading.private_threadpool_size = config.data.num_threads
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
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda image, _: generate_sarsa_opt_trajectory(image=image,
                                                                   reward=config.data.reward_final,
                                                                   gamma=config.data.gamma,
                                                                   min_num_steps=config.data.min_traj_len,
                                                                   max_num_steps=config.data.max_traj_len,
                                                                   with_reversed_actions=config.data.with_reversed_actions,
                                                                   with_random_actions=config.data.with_random_actions),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda transitions, rewards: tf.data.Dataset.from_tensor_slices((transitions, rewards)))
        if not evaluation:
            ds = ds.shuffle(shuffle_buffer_size) # перемешивание
        for batch_size in reversed(batch_dims):
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size) # загружает данные в фоновом режиме

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds, dataset_builder
