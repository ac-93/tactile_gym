from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.example_envs.example_arm_env.example_arm_env import ExampleArmEnv


def main():

    seed = int(0)
    num_iter = 10
    max_steps = 10000
    show_gui = True
    show_tactile = False
    render = False
    print_info = True
    image_size = [256, 256]
    env_modes = {
        # which dofs can have movement (environment dependent)
        "movement_mode": "xyzRxRyRz",

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "franka_panda",
        # "arm_type": "kuka_iiwa",
        # "arm_type": "mg400",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # which observation type to return
        "observation_mode": "oracle",
        # 'observation_mode':'tactile',
        # 'observation_mode':'visual',
        # 'observation_mode':'visuotactile',
        # 'observation_mode':'tactile_and_feature',
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        # the reward type
        # 'reward_mode':'sparse'
        "reward_mode": "dense",
    }

    env = ExampleArmEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seed for deterministic results
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    # create controllable parameters on GUI
    action_ids = []
    min_action, max_action = env.min_action, env.max_action
    if show_gui:
        action_ids.append(
            env._pb.addUserDebugParameter("dx", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dy", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dz", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dRX", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dRY", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dRZ", min_action, max_action, 0)
        )

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    main()
