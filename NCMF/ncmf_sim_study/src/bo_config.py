# List of possible hyperparameters that can be searched using Bayesian Optimization and their constraints

#WARNING: Do not change the order of the below hyperparameters. You may however modify the domain/type as required.

bounds = [{"name": "learning_rate", "type": "continuous", "domain": (1e-6,1e-4)},
    {"name": "convg_thres", "type": "continuous", "domain": (1e-5,1e-4)},
    {"name": "weight_decay", "type": "continuous", "domain": (0.05,0.5)},
    {"name": "kf", "type": "continuous", "domain": (0.1,0.5)},
    {"name": "k", "type": "discrete", "domain": (10,100,200)},
    {"name": "num_chunks", "type": "discrete", "domain": (1,2)},
    {"name": "pretrain_thres", "type": "continuous", "domain": (1e-5,1e-4)}]