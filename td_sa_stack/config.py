import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.wandb_track = True
    config.multi_device = True

    # data
    config.data = data = ml_collections.ConfigDict()
    data.num_threads = 48
    data.min_traj_len = 5
    data.max_traj_len = 25
    data.gamma = 0.9
    data.reward_final = 10
    data.random_flip = True
    data.uniform_dequantization = True
    data.with_reversed_actions = True
    data.with_random_actions = True

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'RegressionInceptionNetV1'
    model.activation = "swish"
    model.optimizer = "adamw"
    model.optimizer_weight_decay = 1e-5

    # training
    config.train = train = ml_collections.ConfigDict()
    train.n_steps = 500_000
    train.lr = 1e-4
    train.batch_size = 1024 # 512 for each device
    train.ema = 0.99

    train.update_target_every = 20
    train.log_every = 200
    train.save_every = 1_000

    # evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 20


    return config