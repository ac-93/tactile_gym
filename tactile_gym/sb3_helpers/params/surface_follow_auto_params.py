import torch.nn as nn
import kornia.augmentation as K
from stable_baselines3.common.torch_layers import NatureCNN
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor

# ============================== RAD ==============================
augmentations = nn.Sequential(
    K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),
)

# ============================== PPO ==============================
rl_params_ppo = {
    # ==== env params ====
    "algo_name": "ppo",
    "env_name": "surface_follow-v0",
    "max_ep_len": 200,
    "image_size": [128, 128],
    "env_modes": {
        ## which dofs can have movement
        # 'movement_mode':'yz',
        # 'movement_mode':'xyz',
        # 'movement_mode':'yzRx',
        "movement_mode": "xyzRxRy",

        # specify arm and tactile sensor
        "arm_type": "ur5",
        # "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digitac",
        "tactile_sensor_name": "digit",

        ## the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        ## noise params for additional robustness
        # 'noise_mode':'none',
        "noise_mode": "simplex",
        # 'noise_mode':'random',

        ## which observation type to return
        # "observation_mode": "oracle",
        "observation_mode": "tactile",
        # 'observation_mode':'visual',
        # 'observation_mode':'visuotactile',

        ## which reward type to use (currently only dense)
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    },
    # ==== control params ====
    'policy': 'MultiInputPolicy',
    "seed": int(1),
    "n_stack": 1,
    "total_timesteps": int(5e5),
    "n_eval_episodes": 10,
    "n_envs": 10,
    "eval_freq": 2e3,
}

ppo_params = {
    # === net arch ===
    "policy_kwargs": {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            'cnn_base': NatureCNN,
            'cnn_output_dim': 256,
            'mlp_extractor_net_arch': [64, 64],
        },
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "activation_fn": nn.Tanh,
    },

    # ==== rl params ====
    "learning_rate": 3e-4,
    "n_steps": int(2048),
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.9,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": 0.1,
}


# ============================== SAC ==============================

rl_params_sac = {
    # ==== env params ====
    "algo_name": "sac",
    "env_name": "surface_follow-v0",
    "max_ep_len": 200,
    "image_size": [128, 128],
    "env_modes": {
        ## which dofs can have movement
        # 'movement_mode':'yz',
        # 'movement_mode':'xyz',
        # 'movement_mode':'yzRx',
        "movement_mode": "xyzRxRy",

        ## the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        ## noise params for additional robustness
        # 'noise_mode':'none',
        "noise_mode": "simplex",
        # 'noise_mode':'random',

        ## which observation type to return
        # "observation_mode": "oracle",
        "observation_mode": "tactile",
        # 'observation_mode':'visual',
        # 'observation_mode':'visuotactile',

        ## which reward type to use (currently only dense)
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    },
    # ==== control params ====
    'policy': 'MultiInputPolicy',
    "n_stack": 1,
    "seed": int(1),
    "total_timesteps": int(1e6),
    "n_eval_episodes": 10,
    "n_envs": 1,
    "eval_freq": 1e4,
}


sac_params = {
    # === net arch ===
    "policy_kwargs": {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            'cnn_base': NatureCNN,
            'cnn_output_dim': 256,
            'mlp_extractor_net_arch': [64, 64],
        },
        "net_arch": dict(pi=[256, 256], qf=[256, 256]),
        "activation_fn": nn.Tanh,
    },

    # ==== rl params ====
    "learning_rate": 3e-4,
    "buffer_size": int(2e4),
    "learning_starts": 1e4,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "action_noise": None,
    "optimize_memory_usage": False,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
}
