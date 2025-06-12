import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    # data
    config.data = data = ml_collections.ConfigDict()
    data.min_traj_len = 60
    data.max_traj_len = 100
    data.gamma = 0.99
    data.reward_final = 10

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'RegressionInceptionNetV1'
    model.activation = "swish"
    model.optimizer = "adamw"
    model.optimizer_hparams = {"lr": 1e-4,
                               "weight_decay": 1e-5}

    # training
    config.train = train = ml_collections.ConfigDict()
    train.n_steps = 500_000
    train.lr = 1e-4
    train.batch_size = 128
    train.ema = 0.99

    train.update_target_every = 40
    train.log_every = 50
    train.save_every = 50_000

    # evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 20


    return config