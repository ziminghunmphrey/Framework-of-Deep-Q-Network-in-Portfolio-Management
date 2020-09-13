abspath=""                 # the file path where you want to save the parameter of the model
tuned_config = {"env": {"window_length":10,
                        "trading_period":2,
                        "expan_coe":200,
                        "norm": 'latest_close',
                        "trading_cost":0,    # Based on the assumptions in the paper, commision fee is set as 0
                        "asset_num":6,       # 5 stocks and 1 cash
                        "feature_num":4},    # open price, close price, highest price, lowest price
                "train": {"learning_rate":0.00025,
                          "division": 5,     # we divide the total portfolio value into 5 equal parts
                          "epsilon": 1,
                          "epsilon_decay_period":15000,
                          "start_date":2016-12,
                          "end_date":2017-12,
                          "date_range_s":20161231,
                          "date_range_e":20171231,
                          "steps_per_episode":160,
                          "reward_scale": 1,
                          "batch_size": 100,
                          "steps": 40000,    # training step
                          "replay_period": 10,
                          "memory_size": 3000,
                          "upd_tar_prd": 200,
                          "save": True,
                          "save_period": 5000,  # save parameters every 5000 steps
                          "discount": 0.17},
                "net": {"kernels": [[1, 3], [1, 5], [1, 1]],
                        "strides": [[1, 2], [1, 5], [1, 1]],
                        "filters": [32,64,128],
                        "padding": "same",
                        "regularizer": 1e-4,
                        "b_initializer": 0.01,
                        "w_initializer": 0.01,
                        "cnn_activation": "selu",
                        "fc_activation": "selu",
                        "fc1_size": 512,
                        "output_num": 462}}
