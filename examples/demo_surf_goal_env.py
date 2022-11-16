from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.exploration.surface_follow.surface_follow_goal.surface_follow_goal_env import (
    SurfaceFollowGoalEnv,
)


def main():

    seed = int(0)
    num_iter = 10
    max_steps = 10000
    show_gui = True
    show_tactile = False
    render = True
    print_info = False
    image_size = [256, 256]
    env_modes = {
        # which dofs can have movement
        # 'movement_mode':'yz',
        # 'movement_mode':'xyz',
        # 'movement_mode':'yzRx',
        "movement_mode": "xyzRxRy",

        # specify arm
        "arm_type": "ur5",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # noise params for additional robustness
        # 'noise_mode':'none',
        # 'noise_mode':'random',
        "noise_mode": "simplex",

        # which observation type to return
        'observation_mode': 'oracle',
        # "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        # which reward type to use (currently only dense)
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = SurfaceFollowGoalEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seeding
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    # create controllable parameters on GUI
    action_ids = []
    min_action = env.min_action
    max_action = env.max_action

    if show_gui:
        if env_modes["movement_mode"] == "yz":
            action_ids.append(env._pb.addUserDebugParameter("dY", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dZ", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "xyz":
            action_ids.append(env._pb.addUserDebugParameter("dX", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dY", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dZ", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "yzRx":
            action_ids.append(env._pb.addUserDebugParameter("dY", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dZ", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRx", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "xyzRxRy":
            action_ids.append(env._pb.addUserDebugParameter("dX", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dY", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dZ", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRy", min_action, max_action, 0))

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    main()
