from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.exploration.edge_follow.edge_follow_env import EdgeFollowEnv


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
        ## which dofs can have movement
        #"movement_mode": "xy",
        "movement_mode": "xyz",

        ## the type of control used
         "control_mode": "TCP_position_control",
        #'control_mode':'TCP_velocity_control',

        # add variation to embed distance to optimise for
        # warning, don't use rand height when controlling z unless
        # including embed distance in observation
        # 'noise_mode':'fixed_height',
        "noise_mode": "rand_height",

        ## which observation type to return
        'observation_mode':'oracle',
        # "observation_mode": "tactile",
        # 'observation_mode':'visual',
        # 'observation_mode':'visuotactile',

        ## which reward type to use (currently only dense)
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = EdgeFollowEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size
    )

    # set seed for deterministic results
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    # create controllable parameters on GUI
    action_ids = []
    min_action = env.min_action
    max_action = env.max_action
    if show_gui:

        if env_modes["movement_mode"] == "xy":
            action_ids.append(
                env._pb.addUserDebugParameter("dX", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dY", min_action, max_action, 0)
            )

        elif env_modes["movement_mode"] == "xyz":
            action_ids.append(
                env._pb.addUserDebugParameter("dX", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dY", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dZ", min_action, max_action, 0)
            )

        elif env_modes["movement_mode"] == "xyRz":
            action_ids.append(
                env._pb.addUserDebugParameter("dX", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dY", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dRz", min_action, max_action, 0)
            )

        elif env_modes["movement_mode"] == "xyzRz":
            action_ids.append(
                env._pb.addUserDebugParameter("dX", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dY", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dZ", min_action, max_action, 0)
            )
            action_ids.append(
                env._pb.addUserDebugParameter("dRz", min_action, max_action, 0)
            )

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    main()
