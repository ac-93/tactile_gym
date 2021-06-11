import os
import gym
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env


def make_training_envs(env_id, rl_params, save_dir):

    env = make_vec_env(
        env_id,
        n_envs=rl_params["n_envs"],
        seed=rl_params["seed"],
        vec_env_cls=SubprocVecEnv,
        monitor_dir=save_dir,
        env_kwargs={
            "show_gui": False,
            "show_tactile": False,
            "max_steps": rl_params["max_ep_len"],
            "image_size": rl_params["image_size"],
            "env_modes": rl_params["env_modes"],
        },
    )
    # stack the images for frame history
    env = VecFrameStack(env, n_stack=rl_params["n_stack"])

    # transpose images in observation
    env = VecTransposeImage(env)

    return env


def make_eval_env(
    env_name,
    rl_params,
    show_gui=False,
    show_tactile=False,
):
    """
    Make a single environment with visualisation specified.
    """
    eval_env = gym.make(
        env_name,
        max_steps=rl_params["max_ep_len"],
        image_size=rl_params["image_size"],
        env_modes=rl_params["env_modes"],
        show_gui=show_gui,
        show_tactile=show_tactile,
    )

    # wrap in monitor
    eval_env = Monitor(eval_env)

    # dummy vec env generally faster than SubprocVecEnv for small networks
    eval_env = DummyVecEnv([lambda: eval_env])

    # stack observations
    eval_env = VecFrameStack(eval_env, n_stack=rl_params["n_stack"])

    # transpose images in observation
    eval_env = VecTransposeImage(eval_env)

    return eval_env
