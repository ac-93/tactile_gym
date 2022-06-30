import sys


def import_parameters(env_name, algo_name):

    if env_name == "edge_follow-v0":
        from tactile_gym.sb3_helpers.params.edge_follow_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
            augmentations,
        )

    elif env_name == "surface_follow-v0":
        from tactile_gym.sb3_helpers.params.surface_follow_auto_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
            augmentations,
        )

    elif env_name == "surface_follow-v1":
        from tactile_gym.sb3_helpers.params.surface_follow_goal_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
            augmentations,
        )
    elif env_name == "surface_follow-v2":
        from tactile_gym.sb3_helpers.params.surface_follow_vert_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
            augmentations,
        )
    elif env_name == "object_roll-v0":
        from tactile_gym.sb3_helpers.params.object_roll_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
            augmentations,
        )

    elif env_name == "object_push-v0":
        from tactile_gym.sb3_helpers.params.object_push_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
            augmentations,
        )

    elif env_name == "object_balance-v0":
        from tactile_gym.sb3_helpers.params.object_balance_params import (
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
            augmentations,
        )

    else:
        sys.exit("Incorrect environment specified: {}.".format(env_name))

    if "ppo" in algo_name:
        return rl_params_ppo, ppo_params, augmentations
    elif "sac" in algo_name:
        return rl_params_sac, sac_params, augmentations
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))
