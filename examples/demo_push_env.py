from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)


def main():

    seed = int(0)
    num_iter = 10
    max_steps = 10000
    show_gui = True
    show_tactile = True
    render = False
    print_info = False
    image_size = [128, 128]
    env_modes = {
        # which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        # "movement_mode": "xyRz",
        'movement_mode': 'TyRz',
        # 'movement_mode':'TxTyRz',

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "mg400",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # randomisations
        "rand_init_orn": False,
        "rand_obj_mass": False,

        # straight or random trajectory
        # "traj_type": "straight",
        'traj_type': 'simplex',

        # which observation type to return
        # 'observation_mode':'oracle',
        "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = ObjectPushEnv(
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
        if env_modes["movement_mode"] == "y":
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))

        if env_modes["movement_mode"] == "yRz":
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRz", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "xyRz":
            action_ids.append(env._pb.addUserDebugParameter("dx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRz", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "TyRz":
            action_ids.append(env._pb.addUserDebugParameter("dTy", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRz", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "TxTyRz":
            action_ids.append(env._pb.addUserDebugParameter("dTx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dTy", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRz", min_action, max_action, 0))

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    main()
