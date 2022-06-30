import gym
import os
import sys
import time
import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps, CheckpointCallback

from stable_baselines3 import PPO, SAC
from sb3_contrib import RAD_SAC, RAD_PPO

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.sb3_helpers.rl_utils import make_training_envs, make_eval_env
from tactile_gym.sb3_helpers.eval_agent_utils import final_evaluation
from tactile_gym.utils.general_utils import (
    save_json_obj,
    load_json_obj,
    convert_json,
    check_dir,
)
from tactile_gym.sb3_helpers.custom.custom_callbacks import (
    FullPlottingCallback,
    ProgressBarManager,
)
import argparse

import torch.nn as nn
import kornia.augmentation as K
from stable_baselines3.common.torch_layers import NatureCNN
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor

# ============================== RAD ==============================
augmentations = nn.Sequential(
    K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),
)


parser = argparse.ArgumentParser(description="Train an agent in a tactile gym task.")
# metavar ='' can tidy the help tips.
# parser.add_argument("-E", '--env_name', type=str, required = True, help='The name of a tactile gym env.', metavar='')
parser.add_argument("-M", '--movement_mode', type=str, help='The movement mode.', metavar='')
parser.add_argument("-T", '--traj_type', type=str, help='The traj type.', metavar='')
parser.add_argument("-R", '--retrain_path', type=str, help='Retrain model path.', metavar='')
parser.add_argument("-I", '--if_retrain', type=str, help='Retrain.', metavar='')


args = parser.parse_args()


def fix_floats(data):
    if isinstance(data, list):
        iterator = enumerate(data)
    elif isinstance(data, dict):
        iterator = data.items()
    else:
        raise TypeError("can only traverse list or dict")

    for i, value in iterator:
        if isinstance(value, (list, dict)):
            fix_floats(value)
        elif isinstance(value, str):
            try:
                data[i] = float(value)
            except ValueError:
                pass


def train_agent(
    algo_name="ppo",
    env_name="edge_follow-v0",
    rl_params={},
    algo_params={},
    augmentations=None,
):

    # create save dir
    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_dir = os.path.join(
        "saved_models/", rl_params["env_name"], timestr, algo_name, "s{}_{}".format(
            rl_params["seed"], rl_params["env_modes"]["observation_mode"])
    )

    check_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # save params
    save_json_obj(convert_json(rl_params), os.path.join(save_dir, "rl_params"))
    save_json_obj(convert_json(algo_params), os.path.join(save_dir, "algo_params"))
    if "rad" in algo_name:
        save_json_obj(convert_json(augmentations), os.path.join(save_dir, "augmentations"))

    # load the envs
    env = make_training_envs(env_name, rl_params, save_dir)

    eval_env = make_eval_env(
        env_name,
        rl_params,
        show_gui=False,
        show_tactile=False,
    )

    # define callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "trained_models/"),
        log_path=os.path.join(save_dir, "trained_models/"),
        eval_freq=rl_params["eval_freq"],
        n_eval_episodes=rl_params["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    plotting_callback = FullPlottingCallback(log_dir=save_dir, total_timesteps=rl_params["total_timesteps"])
    event_plotting_callback = EveryNTimesteps(n_steps=rl_params["eval_freq"] * rl_params["n_envs"], callback=plotting_callback)

    # create the model with hyper params
    if algo_name == "ppo":
        model = PPO(rl_params["policy"], env, **algo_params, verbose=1)
    elif algo_name == "rad_ppo":
        model = RAD_PPO(rl_params["policy"], env, **algo_params, augmentations=augmentations, visualise_aug=False, verbose=1)
    elif algo_name == "sac":
        model = SAC(rl_params["policy"], env, **algo_params, verbose=1)
    elif algo_name == "rad_sac":
        model = RAD_SAC(rl_params["policy"], env, **algo_params, augmentations=augmentations, visualise_aug=False, verbose=1)
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))

    # train an agent
    with ProgressBarManager(rl_params["total_timesteps"]) as progress_bar_callback:
        model.learn(
            total_timesteps=rl_params["total_timesteps"],
            callback=[progress_bar_callback, eval_callback, event_plotting_callback],
        )

    # save the final model after training
    model.save(os.path.join(save_dir, "trained_models", "final_model"))
    env.close()
    eval_env.close()

    # run final evaluation over 20 episodes and save a vid
    final_evaluation(
        saved_model_dir=save_dir,
        n_eval_episodes=10,
        seed=None,
        deterministic=True,
        show_gui=False,
        show_tactile=False,
        render=True,
        save_vid=True,
        take_snapshot=False,
    )


def retrain_agent(model_path,
                  algo_name='ppo',
                  env_name='edge_follow-v0',
                  rl_params={},
                  algo_params={},
                  augmentations=None,):

    timestr = time.strftime("%Y%m%d-%H%M%S")

    new_save_dir = os.path.join(
        "saved_models/", "retrain_models/", env_name, timestr, algo_name, "s{}_{}".format(
            rl_params["seed"], rl_params["env_modes"]["observation_mode"])
    )
    check_dir(new_save_dir)
    os.makedirs(new_save_dir, exist_ok=True)
    # save params
    save_json_obj(convert_json(rl_params), os.path.join(new_save_dir, "rl_params"))
    save_json_obj(convert_json(algo_params), os.path.join(new_save_dir, "algo_params"))
    if 'rad' in algo_name:
        save_json_obj(convert_json(augmentations), os.path.join(new_save_dir, "augmentations"))
    # load the envs
    env = make_training_envs(
        env_name,
        rl_params,
        new_save_dir
    )

    eval_env = make_eval_env(
        env_name,
        rl_params,
        show_gui=False,
        show_tactile=False,
    )

    # define callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(new_save_dir, "trained_models/"),
        log_path=os.path.join(new_save_dir, "trained_models/"),
        eval_freq=rl_params["eval_freq"],
        n_eval_episodes=rl_params["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    plotting_callback = FullPlottingCallback(log_dir=new_save_dir, total_timesteps=rl_params['total_timesteps'])
    event_plotting_callback = EveryNTimesteps(n_steps=rl_params['eval_freq']*rl_params['n_envs'], callback=plotting_callback)

    save_frequency = 400000
    checkpoint_callback = CheckpointCallback(save_freq=save_frequency / rl_params["n_envs"], save_path=os.path.join(new_save_dir, "trained_models/"),
                                             name_prefix='rl_model')
    # creat agent and load the policy zip file
    if algo_name == 'ppo':
        model = PPO(

            rl_params["env_name"],
            env,
            **algo_params,
            verbose=1
        )
    elif algo_name == 'rad_ppo':
        model = RAD_PPO(
            rl_params["policy"],
            env,
            **algo_params,
            augmentations=augmentations,
            visualise_aug=False,
            verbose=1
        )

        model = model.load(model_path, env=env)
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))
    # set_trace()
    # train an agent
    with ProgressBarManager(
        rl_params["total_timesteps"]
    ) as progress_bar_callback:
        model.learn(
            total_timesteps=rl_params["total_timesteps"],
            callback=[progress_bar_callback, eval_callback, event_plotting_callback, checkpoint_callback],
        )

    # save the final model after training
    model.save(os.path.join(new_save_dir, "trained_models", "final_model"))
    env.close()
    eval_env.close()

    # run final evaluation over 20 episodes and save a vid
    final_evaluation(
        saved_model_dir=new_save_dir,
        n_eval_episodes=10,
        seed=None,
        deterministic=True,
        show_gui=False,
        show_tactile=False,
        render=True,
        save_vid=True,
        take_snapshot=False
    )


if __name__ == "__main__":

    if args.if_retrain:
        saved_model_dir = args.retrain_path
        model_path = os.path.join(saved_model_dir, "trained_models", "best_model.zip")
        rl_params = load_json_obj(os.path.join(saved_model_dir, "rl_params"))
        algo_params = load_json_obj(os.path.join(saved_model_dir, "algo_params"))

        env_name = rl_params["env_name"]
        algo_name = rl_params["algo_name"]
        # need to load the class
        if algo_params['policy_kwargs']['features_extractor_class'] == "CustomCombinedExtractor":
            algo_params['policy_kwargs']['features_extractor_class'] = CustomCombinedExtractor
        if algo_params['policy_kwargs']['features_extractor_kwargs']['cnn_base'] == "NatureCNN":
            algo_params['policy_kwargs']['features_extractor_kwargs']['cnn_base'] = NatureCNN
        if algo_params['policy_kwargs']['activation_fn'] == "Tanh":
            algo_params['policy_kwargs']['activation_fn'] = nn.Tanh
        if os.path.isfile(os.path.join(saved_model_dir, 'augmentations.json')):
            algo_name = "rad_" + algo_name
            augmentations = nn.Sequential(K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),)

        retrain_agent(
            model_path,
            algo_name,
            env_name,
            rl_params,
            algo_params,
            augmentations
        )
    else:
        # choose which RL algo to use
        # algo_name = 'ppo'
        algo_name = "rad_ppo"
        # algo_name = 'sac'
        # algo_name = 'rad_sac'
    #
        env_name = "edge_follow-v0"
        # env_name = 'surface_follow-v2'
        # env_name = 'surface_follow-v1'
        # env_name = 'surface_follow-v0'
        # env_name = 'object_roll-v0'
        # env_name = "object_push-v0"
        # env_name = 'object_balance-v0'

        # import paramters
        rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)
        train_agent(
            algo_name,
            env_name,
            rl_params,
            algo_params,
            augmentations
        )
