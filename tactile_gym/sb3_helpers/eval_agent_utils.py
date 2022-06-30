import os
import sys
import numpy as np
import cv2

import stable_baselines3 as sb3

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.rl_utils import make_eval_env
from tactile_gym.utils.general_utils import load_json_obj


def eval_and_save_vid(
    model, env, saved_model_dir, n_eval_episodes=10, deterministic=True, render=False, save_vid=False, take_snapshot=False
):

    if save_vid:
        record_every_n_frames = 1
        render_img = env.render(mode="rgb_array")
        render_img_size = (render_img.shape[1], render_img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(saved_model_dir, "evaluated_policy.mp4"),
            fourcc,
            24.0,
            render_img_size,
        )

    if take_snapshot:
        render_img = env.render(mode="rgb_array")
        render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(saved_model_dir, "env_snapshot.png"), render_img)

    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0

        while not done:

            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # render visual + tactile observation
            if render:
                render_img = env.render(mode="rgb_array")
            else:
                render_img = None

            # write rendered image to mp4
            # use record_every_n_frames to reduce size sometimes
            if save_vid and episode_length % record_every_n_frames == 0:

                # warning to enable rendering
                if render_img is None:
                    sys.exit("Must be rendering to save video")

                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            if take_snapshot:
                render_img = env.render(mode="rgb_array")
                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(saved_model_dir, "env_snapshot.png"), render_img)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    if save_vid:
        out.release()

    return episode_rewards, episode_lengths


def final_evaluation(
    saved_model_dir,
    n_eval_episodes,
    seed=None,
    deterministic=True,
    show_gui=True,
    show_tactile=True,
    render=False,
    save_vid=False,
    take_snapshot=False,
):

    rl_params = load_json_obj(os.path.join(saved_model_dir, "rl_params"))
    algo_params = load_json_obj(os.path.join(saved_model_dir, "algo_params"))

    print(rl_params['env_name'])
    print(rl_params['env_modes'])
    # create the evaluation env
    eval_env = make_eval_env(
        rl_params["env_name"],
        rl_params,
        show_gui=show_gui,
        show_tactile=show_tactile,
    )

    # load the trained model
    model_path = os.path.join(saved_model_dir, "trained_models", "best_model.zip")
    # model_path = os.path.join(saved_model_dir, "trained_models", "final_model.zip")

    # create the model with hyper params
    if rl_params["algo_name"] == "ppo":
        model = sb3.PPO.load(model_path)
    elif rl_params["algo_name"] == "sac":
        model = sb3.SAC.load(model_path)
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))

    # seed the env
    if seed is not None:
        eval_env.reset()
        eval_env.seed(seed)

    # evaluate the trained agent
    episode_rewards, episode_lengths = eval_and_save_vid(
        model,
        eval_env,
        saved_model_dir,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        save_vid=save_vid,
        render=render,
        take_snapshot=take_snapshot,
    )

    print("Avg Ep Rew: {}, Avg Ep Len: {}".format(np.mean(episode_rewards), np.mean(episode_lengths)))

    eval_env.close()


if __name__ == "__main__":

    # evaluate params
    n_eval_episodes = 10
    seed = int(1)
    deterministic = True
    show_gui = True
    show_tactile = True
    render = False
    save_vid = False
    take_snapshot = False

    ## load the trained model
    # algo_name = 'ppo'
    algo_name = "rad_ppo"
    # algo_name = 'sac'
    # algo_name = 'rad_sac'

    # env_name = 'edge_follow-v0'
    # env_name = "surface_follow-v0"
    env_name = "surface_follow-v1"
    # env_name = 'surface_follow-v2'
    # env_name = 'object_roll-v0'
    # env_name = 'object_push-v0'
    # env_name = 'object_balance-v0'

    # obs_type = 'oracle'
    # obs_type = "s1_tactile"
    obs_type = "s1_tactile_and_feature"
    # obs_type = 'visual'
    # obs_type = 'visuotactile'

    ## combine args
    saved_model_dir = os.path.join(os.path.dirname(__file__), "saved_models", env_name, algo_name, obs_type)

    final_evaluation(
        saved_model_dir,
        n_eval_episodes,
        seed=seed,
        deterministic=deterministic,
        show_gui=show_gui,
        show_tactile=show_tactile,
        render=render,
        save_vid=save_vid,
        take_snapshot=take_snapshot,
    )
