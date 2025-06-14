import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.wandb_track = False
    config.multi_device = True

    # data
    config.data = data = ml_collections.ConfigDict()
    data.num_threads = 48
    data.min_traj_len = 40
    data.max_traj_len = 80
    data.gamma = 0.97
    data.reward_final = 10
    data.random_flip = True
    data.with_reversed_actions = True

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
    train.ema = 0.995

    train.update_target_every = 50
    train.log_every = 200
    train.save_every = 1_000

    # evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 20


    return config