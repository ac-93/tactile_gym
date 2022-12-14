import os
import sys
import gym
import numpy as np

from tactile_gym.robots.arms.robot import Robot
from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv
from tactile_gym.rl_envs.example_envs.example_arm_env.rest_poses import rest_poses_dict

env_modes_default = {
    "movement_mode": "xyzRxRyRz",
    "control_mode": "TCP_velocity_control",
    "observation_mode": "oracle",
    "reward_mode": "dense",
}


class ExampleArmEnv(BaseTactileEnv):
    def __init__(
        self,
        max_steps=1000,
        image_size=[64, 64],
        env_modes=dict(),
        show_gui=False,
        show_tactile=False,
    ):

        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(np.floor(self._control_rate / self._sim_time_step))
        self._max_blocking_pos_move_steps = 10

        super(ExampleArmEnv, self).__init__(max_steps, image_size, show_gui, show_tactile)

        # set modes from algorithm side
        self.movement_mode = env_modes["movement_mode"]
        self.control_mode = env_modes["control_mode"]
        self.observation_mode = env_modes["observation_mode"]
        self.reward_mode = env_modes["reward_mode"]

        # set which robot arm and sensor to use
        self.arm_type = env_modes["arm_type"]
        self.t_s_name = env_modes["tactile_sensor_name"]
        self.t_s_type = "standard"
        self.t_s_core = "no_core"

        # setup variables
        self.setup_action_space()

        # load environment objects
        self.load_environment()

        # limits
        TCP_lims = np.zeros(shape=(6, 2))
        TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, +0.1  # x lims
        TCP_lims[1, 0], TCP_lims[1, 1] = -0.1, +0.1  # y lims
        TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, +0.1  # z lims
        TCP_lims[3, 0], TCP_lims[3, 1] = -np.pi / 8, np.pi / 8  # roll lims
        TCP_lims[4, 0], TCP_lims[4, 1] = -np.pi / 8, np.pi / 8  # pitch lims
        TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi / 8, np.pi / 8  # yaw lims

        # set workframe
        self.workframe_pos = np.array([0.65, 0.0, 0.05])
        self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_name][self.t_s_type]

        # load the ur5 with a tactip attached
        self.robot = Robot(
            self._pb,
            rest_poses=rest_poses,
            workframe_pos=self.workframe_pos,
            workframe_rpy=self.workframe_rpy,
            TCP_lims=TCP_lims,
            image_size=image_size,
            turn_off_border=False,
            arm_type=self.arm_type,
            t_s_name=self.t_s_name,
            t_s_type=self.t_s_type,
            t_s_core=self.t_s_core,
            t_s_dynamics={'stiffness': 50, 'damping': 100, 'friction': 10.0},
            show_gui=self._show_gui,
            show_tactile=self._show_tactile,
        )

        # this is needed to set some variables used for initial observation/obs_dim()
        self.reset()

        # set the observation space dependent on
        self.setup_observation_space()

    def setup_action_space(self):

        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action = -0.01, 0.01

        # define action ranges per act dim to rescale output of policy
        if self.control_mode == "TCP_position_control":

            max_pos_change = 0.001  # m per step
            max_ang_change = 1 * (np.pi / 180)  # rad per step

            self.x_act_min, self.x_act_max = -max_pos_change, max_pos_change
            self.y_act_min, self.y_act_max = -max_pos_change, max_pos_change
            self.z_act_min, self.z_act_max = -max_pos_change, max_pos_change
            self.roll_act_min, self.roll_act_max = -max_ang_change, max_ang_change
            self.pitch_act_min, self.pitch_act_max = -max_ang_change, max_ang_change
            self.yaw_act_min, self.yaw_act_max = -max_ang_change, max_ang_change

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
            self.roll_act_min, self.roll_act_max = -max_ang_vel, max_ang_vel
            self.pitch_act_min, self.pitch_act_max = -max_ang_vel, max_ang_vel
            self.yaw_act_min, self.yaw_act_max = -max_ang_vel, max_ang_vel

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def setup_rgb_obs_camera_params(self):
        """
        Setup camera parameters for debug visualiser and RGB observation
        """
        self.rgb_cam_pos = [0.35, 0.0, -0.25]
        self.rgb_cam_dist = 1.0
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -35
        self.rgb_image_size = self._image_size
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100

    def reset(self):

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevalent when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()

        self.reset_counter += 1
        self._env_step_counter = 0

        # reset TCP pos and rpy in work frame
        self.robot.reset(reset_TCP_pos=[0, 0, 0], reset_TCP_rpy=[0, 0, 0])

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def full_reset(self):
        self._pb.resetSimulation()
        self.load_environment()
        self.robot.full_reset()
        self.reset_counter = 0

    def encode_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm
        i.e. NN could be predicting [y, Rz] actions. Make sure they are in
        correct place [1, 5].
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "xyzRxRyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
            encoded_actions[3] = actions[3]
            encoded_actions[4] = actions[4]
            encoded_actions[5] = actions[5]

        else:
            sys.exit("Incorrect movement mode specified: {}".format(self.movement_mode))

        return encoded_actions

    def get_step_data(self):

        # get rl info
        done = self.termination()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward()

        return reward, done

    def termination(self):
        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True
        return False

    def sparse_reward(self):
        return 0.0

    def dense_reward(self):
        return 0.0

    def get_act_dim(self):
        if self.movement_mode == "xyzRxRyRz":
            return 6
        else:
            sys.exit("Incorrect movement mode specified: {}".format(self.movement_mode))
