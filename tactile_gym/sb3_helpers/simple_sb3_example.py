import gym
import torch as th
import kornia.augmentation as K

from stable_baselines3 import PPO, SAC
from sb3_contrib import RAD_SAC, RAD_PPO

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor

if __name__ == "__main__":

    # algo_name = 'ppo'
    # algo_name = 'sac'
    algo_name = "rad_ppo"
    # algo_name = 'rad_sac'

    # show gui can only be enabled for n_envs = 1
    # if using image observation SubprocVecEnv is needed to replace DummyVecEnv
    # as pybullet EGL rendering requires separate processes to avoid silent
    # rendering issues.
    seed = 1
    n_envs = 1
    show_gui = False
    show_tactile = False

    env_modes_default = {
        "movement_mode": "xy",
        "control_mode": "TCP_velocity_control",
        "rand_init_obj_pos": False,
        "rand_obj_size": False,
        "rand_embed_dist": False,
        "observation_mode": "tactile_and_feature",
        "reward_mode": "dense",
    }

    # env_id = "edge_follow-v0"
    # env_id = "surface_follow-v0"
    # env_id = "surface_follow-v1"
    env_id = "object_roll-v0"
    # env_id = "object_push-v0"
    # env_id = "object_balance-v0"

    env = make_vec_env(
        env_id,
        env_kwargs={"show_gui": show_gui, "show_tactile": show_tactile, "env_modes": env_modes_default},
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )

    algo_params = {
        "policy_kwargs": {
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "cnn_output_dim": 128,
                "mlp_extractor_net_arch": [64, 64],
            },
            "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
        },
    }

    # define augmentations to apply
    if "rad" in algo_name:
        augmentations = th.nn.Sequential(
            K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),
        )

    if algo_name == "ppo":
        model = PPO("MultiInputPolicy", env, **algo_params, verbose=1)

    elif algo_name == "rad_ppo":
        model = RAD_PPO("MultiInputPolicy", env, **algo_params, augmentations=augmentations, visualise_aug=True, verbose=1)

    elif algo_name == "sac":
        model = SAC(
            "MultiInputPolicy",
            env,
            # **algo_params,
            verbose=1,
        )

    elif algo_name == "rad_sac":
        model = RAD_SAC(
            "MultiInputPolicy",
            env,
            # **algo_params,
            augmentations=augmentations,
            visualise_aug=True,
            verbose=1,
        )

    model.learn(total_timesteps=100000)

    n_eval_episodes = 10
    for i in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

    env.close()
