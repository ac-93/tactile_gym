from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_roll.object_roll_env import (
    ObjectRollEnv,
)


def main():

    seed = int(0)
    num_iter = 10
    max_steps = 200
    show_gui = True
    show_tactile = False
    render = True
    print_info = False
    image_size = [256, 256]
    env_modes = {
        # which dofs can have movement (environment dependent)
        "movement_mode": "xy",

        # specify arm
        "arm_type": "ur5",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",

        # the type of control used
        # "control_mode": "TCP_position_control",
        'control_mode': 'TCP_velocity_control',

        # add variation to joint force for rigid core
        "rand_init_obj_pos": False,
        "rand_obj_size": False,
        "rand_embed_dist": False,

        # which observation type to return
        'observation_mode': 'oracle',
        # "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = ObjectRollEnv(
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
    min_action = env.min_action
    max_action = env.max_action

    if show_gui:
        if env_modes["movement_mode"] == "x":
            action_ids.append(env._pb.addUserDebugParameter("dx", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "xy":
            action_ids.append(env._pb.addUserDebugParameter("dx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "xyz":
            action_ids.append(env._pb.addUserDebugParameter("dx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dz", min_action, max_action, 0))

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    main()
