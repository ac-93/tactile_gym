import numpy as np
from tactile_gym.rl_envs.exploration.surface_follow.base_surface_env import (
    BaseSurfaceEnv,
)

env_modes_default = {
    "movement_mode": "xyzRxRy",
    "control_mode": "TCP_velocity_control",
    "noise_mode": "simplex",
    "observation_mode": "oracle",
    "reward_mode": "dense",
}


class SurfaceFollowGoalEnv(BaseSurfaceEnv):
    def __init__(
        self,
        max_steps=200,
        image_size=[64, 64],
        env_modes=env_modes_default,
        show_gui=False,
        show_tactile=False,
    ):

        super(SurfaceFollowGoalEnv, self).__init__(max_steps, image_size, env_modes, show_gui, show_tactile)

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to ur5
        """
        encoded_actions = np.zeros(6)

        if self.movement_mode == "yz":
            encoded_actions[1] = actions[0]
            encoded_actions[2] = actions[1]
        if self.movement_mode == "xyz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
        if self.movement_mode == "yzRx":
            encoded_actions[1] = actions[0]
            encoded_actions[2] = actions[1]
            encoded_actions[3] = actions[2]
        if self.movement_mode == "xyzRxRy":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
            encoded_actions[3] = actions[3]
            encoded_actions[4] = actions[4]

        return encoded_actions

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        Reward is accumulated during an episode and given when a goal is achieved.
        """

        self.accum_rew += self.dense_reward()

        dist = self.xyz_dist_to_goal()
        if dist < self.termination_dist:
            reward = self.accum_rew
        else:
            reward = 0

        return reward

    def dense_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        W_goal = 1.0
        W_surf = 10.0
        W_norm = 1.0

        # get the distances
        goal_dist = self.xy_dist_to_goal()
        surf_dist = self.z_dist_to_surface()
        cos_dist = self.cos_dist_to_surface_normal()

        # set the reward for aligning to normal as 0 as not possible in this movement mode
        if self.movement_mode in ["yz", "xyz"]:
            W_norm = 0.0

        # sum rewards with multiplicative factors
        reward = -((W_goal * goal_dist) + (W_surf * surf_dist) + (W_norm * cos_dist))

        return reward

    def get_extended_feature_array(self):
        """
        features needed to help complete task.
        Goal pose and current tcp pose.
        """
        # get sim info on TCP
        (
            tcp_pos_workframe,
            _,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        # convert the features into array that matches the image observation shape
        feature_array = np.array([*tcp_pos_workframe, *self.goal_pos_workframe])

        return feature_array

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        if self.movement_mode == "yz":
            return 2
        if self.movement_mode == "xyz":
            return 3
        if self.movement_mode == "yzRx":
            return 3
        if self.movement_mode == "xyzRxRy":
            return 5
