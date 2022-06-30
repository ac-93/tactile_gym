import os, sys
import gym
import numpy as np
import cv2

from tactile_gym.assets import get_assets_path, add_assets_path
from tactile_gym.rl_envs.nonprehensile_manipulation.base_object_env import BaseObjectEnv
from tactile_gym.rl_envs.nonprehensile_manipulation.object_roll.rest_poses import (
    rest_poses_dict,
)

env_modes_default = {
    "movement_mode": "xy",
    "control_mode": "TCP_velocity_control",
    "rand_init_obj_pos": False,
    "rand_obj_size": False,
    "rand_embed_dist": False,
    "observation_mode": "oracle",
    "reward_mode": "dense",
}


class ObjectRollEnv(BaseObjectEnv):
    def __init__(
        self,
        max_steps=1000,
        image_size=[64, 64],
        env_modes=env_modes_default,
        show_gui=False,
        show_tactile=False,
    ):

        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(np.floor(self._control_rate / self._sim_time_step))
        self._max_blocking_pos_move_steps = 10

        # pull params from env_modes specific to push env
        self.rand_init_obj_pos = env_modes["rand_init_obj_pos"]
        self.rand_obj_size = env_modes["rand_obj_size"]
        self.rand_embed_dist = env_modes["rand_embed_dist"]

        # set which robot arm to use
        self.arm_type = env_modes["arm_type"]
        # self.arm_type = "ur5"
        # self.arm_type = "mg400"
        # self.arm_type = 'franka_panda'
        # self.arm_type = 'kuka_iiwa'

        # which t_s to use
        self.t_s_name = env_modes["tactile_sensor_name"]
        # self.t_s_name = 'tactip'
        # self.t_s_name = 'digit'
        self.t_s_type = "flat"
        self.t_s_core = "fixed"
        self.t_s_dynamics = {"stiffness": 10.0, "damping": 100, "friction": 10.0}

        # distance from goal to cause termination
        self.termination_pos_dist = 0.001

        # how much penetration of the tip to optimize for
        # randomly vary this on each episode
        self.embed_dist = 0.0015

        # turn on goal visualisation
        self.visualise_goal = True

        # work frame origin
        self.workframe_pos = np.array([0.65, 0.0, 2 * 0.0025 - self.embed_dist])
        self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

        # limits
        TCP_lims = np.zeros(shape=(6, 2))
        TCP_lims[0, 0], TCP_lims[0, 1] = -0.05, 0.05  # x lims
        TCP_lims[1, 0], TCP_lims[1, 1] = -0.05, 0.05  # y lims
        TCP_lims[2, 0], TCP_lims[2, 1] = -0.01, 0.01  # z lims
        TCP_lims[3, 0], TCP_lims[3, 1] = 0, 0  # roll lims
        TCP_lims[4, 0], TCP_lims[4, 1] = 0, 0  # pitch lims
        TCP_lims[5, 0], TCP_lims[5, 1] = 0, 0  # yaw lims

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_type]

        # init base env
        super(ObjectRollEnv, self).__init__(
            max_steps,
            image_size,
            env_modes,
            TCP_lims,
            rest_poses,
            show_gui,
            show_tactile,
        )

        # this is needed to set some variables used for initial observation/obs_dim()
        self.reset()

        # set the observation space dependent on
        self.setup_observation_space()

    def setup_action_space(self):
        """
        Sets variables used for making network predictions and
        sending correct actions to robot from raw network predictions.
        """
        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action = -0.25, 0.25

        # define action ranges per act dim to rescale output of policy
        if self.control_mode == "TCP_position_control":

            max_pos_change = 0.001  # m per step
            max_ang_change = 1 * (np.pi / 180)  # rad per step

            self.x_act_min, self.x_act_max = -max_pos_change, max_pos_change
            self.y_act_min, self.y_act_max = -max_pos_change, max_pos_change
            self.z_act_min, self.z_act_max = 0, 0
            self.roll_act_min, self.roll_act_max = 0, 0
            self.pitch_act_min, self.pitch_act_max = 0, 0
            self.yaw_act_min, self.yaw_act_max = 0, 0

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = 0, 0
            self.roll_act_min, self.roll_act_max = 0, 0
            self.pitch_act_min, self.pitch_act_max = 0, 0
            self.yaw_act_min, self.yaw_act_max = 0, 0

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def setup_rgb_obs_camera_params(self):
        self.rgb_cam_pos = [0.75, 0.0, 0.00775]
        self.rgb_cam_dist = 0.01
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = 0
        self.rgb_image_size = self._image_size
        # self.rgb_image_size = [512,512]
        self.rgb_fov = 75
        self.rgb_near_val = 0.01
        self.rgb_far_val = 100

    def setup_object(self):
        """
        Set vars for loading an object
        """
        # currently hardcode these for cube, could pull this from bounding box
        self.default_obj_radius = 0.0025

        # define an initial position for the objects (world coords)
        self.init_obj_pos = [0.65, 0.0, self.default_obj_radius]

        self.init_obj_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])

        # textured objects don't render in direct mode
        if self._show_gui:
            self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_roll/sphere/sphere_tex.urdf")
        else:
            self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_roll/sphere/sphere.urdf")

        self.goal_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_roll/sphere/sphere.urdf")

    def reset_task(self):
        """
        Change marble size if enabled.
        Change embed distance if enabled.
        """
        if self.rand_obj_size:
            self.scaling_factor = self.np_random.uniform(1.0, 2.0)
        else:
            self.scaling_factor = 1.0

        self.scaled_obj_radius = self.default_obj_radius * self.scaling_factor

        if self.rand_embed_dist:
            self.embed_dist = self.np_random.uniform(0.0015, 0.003)

    def update_workframe(self):
        """
        Change workframe on reset if needed
        """
        # reset workframe origin based on new obj radius
        self.workframe_pos = np.array([0.65, 0.0, 2 * self.scaled_obj_radius - self.embed_dist])

        # set the arm workframe
        self.robot.arm.set_workframe(self.workframe_pos, self.workframe_rpy)

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        # reset the position of the object
        if self.rand_init_obj_pos:
            self.init_obj_pos = [
                0.65 + self.np_random.uniform(-0.009, 0.009),
                0.0 + self.np_random.uniform(-0.009, 0.009),
                self.scaled_obj_radius,
            ]
        else:
            self.init_obj_pos = [0.65, 0.0, self.scaled_obj_radius]

        self.init_obj_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])

        if not self.rand_obj_size:
            self._pb.resetBasePositionAndOrientation(self.obj_id, self.init_obj_pos, self.init_obj_orn)
        else:
            self._pb.removeBody(self.obj_id)

            self.obj_id = self._pb.loadURDF(
                self.object_path, self.init_obj_pos, self.init_obj_orn, globalScaling=self.scaling_factor
            )

        # could perform object dynamics randomisations here
        self._pb.changeDynamics(
            self.obj_id,
            -1,
            lateralFriction=10.0,
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            frictionAnchor=0,
            collisionMargin=0.000001,
        )

    def make_goal(self):
        """
        Generate a goal place a set distance from the inititial object pose.
        """

        # place goal randomly
        goal_ang = self.np_random.uniform(-np.pi, np.pi)
        if self.rand_init_obj_pos:
            goal_dist = self.np_random.uniform(low=0.0, high=0.015)
        else:
            goal_dist = self.np_random.uniform(low=0.005, high=0.015)

        self.goal_pos_tcp = np.array([goal_dist * np.cos(goal_ang), goal_dist * np.sin(goal_ang), 0.0])

        self.goal_rpy_tcp = [0.0, 0.0, 0.0]
        self.goal_orn_tcp = self._pb.getQuaternionFromEuler(self.goal_rpy_tcp)

        self.update_goal()

    def update_goal(self):
        """
        Transforms goal in TCP frame to a pose in world frame.
        """
        (
            cur_tcp_pos,
            _,
            cur_tcp_orn,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_worldframe()
        (
            self.goal_pos_worldframe,
            self.goal_orn_worldframe,
        ) = self._pb.multiplyTransforms(cur_tcp_pos, cur_tcp_orn, self.goal_pos_tcp, self.goal_orn_tcp)
        self.goal_rpy_worldframe = self._pb.getEulerFromQuaternion(self.goal_orn_worldframe)

        # create variables for goal pose in workframe frame to use later
        (
            self.goal_pos_workframe,
            self.goal_rpy_workframe,
        ) = self.robot.arm.worldframe_to_workframe(self.goal_pos_worldframe, self.goal_rpy_worldframe)
        self.goal_orn_workframe = self._pb.getQuaternionFromEuler(self.goal_rpy_workframe)

        # useful for visualisation
        if self.visualise_goal:
            self._pb.resetBasePositionAndOrientation(self.goal_indicator, self.goal_pos_worldframe, self.goal_orn_worldframe)

    def encode_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm.
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "xy":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]

        return encoded_actions

    def get_step_data(self):

        # update the world position of the goal based on current position of TCP
        self.update_goal()

        # get the cur tip pos here for once per step
        (
            self.cur_tcp_pos_worldframe,
            self.cur_tcp_rpy_worldframe,
            self.cur_tcp_orn_worldframe,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_worldframe()
        (
            self.cur_obj_pos_worldframe,
            self.cur_obj_orn_worldframe,
        ) = self.get_obj_pos_worldframe()

        # get rl info
        done = self.termination()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward()

        return reward, done

    def termination(self):
        """
        Criteria for terminating an episode.
        """
        # terminate when distance to goal is < eps
        pos_dist = self.xy_obj_dist_to_goal()

        if pos_dist < self.termination_pos_dist:
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        +1 is given if object reaches goal.
        """
        # terminate when distance to goal is < eps
        pos_dist = self.xy_obj_dist_to_goal()

        if pos_dist < self.termination_pos_dist:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    def dense_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        W_obj_goal_pos = 1.0

        goal_pos_dist = self.xy_obj_dist_to_goal()

        # sum rewards with multiplicative factors
        reward = -(W_obj_goal_pos * goal_pos_dist)

        return reward

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get sim info on object
        cur_obj_pos_workframe, cur_obj_orn_workframe = self.get_obj_pos_workframe()
        (
            cur_obj_lin_vel_workframe,
            cur_obj_ang_vel_workframe,
        ) = self.get_obj_vel_workframe()

        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            tcp_orn_workframe,
            tcp_lin_vel_workframe,
            tcp_ang_vel_workframe,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        # stack into array
        observation = np.hstack(
            [
                *tcp_pos_workframe,
                *tcp_orn_workframe,
                *tcp_lin_vel_workframe,
                *tcp_ang_vel_workframe,
                *cur_obj_pos_workframe,
                *cur_obj_orn_workframe,
                *cur_obj_lin_vel_workframe,
                *cur_obj_ang_vel_workframe,
                *self.goal_pos_tcp,
                *self.goal_orn_tcp,
                self.scaled_obj_radius,
            ]
        )

        return observation

    def get_extended_feature_array(self):
        """
        features needed to help complete task.
        Goal pose in TCP frame.
        """
        feature_array = np.array([*self.goal_pos_tcp])
        return feature_array

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        if self.movement_mode == "xy":
            return 2

    def overlay_goal_on_image(self, tactile_image):
        """
        Overlay a crosshairs onto the observation in roughly the position
        of the goal
        """
        # get the coords of the goal in image space
        # min/max from 20mm radius tip + extra for border
        min, max = -0.021, 0.021
        norm_tcp_pos_x = (self.goal_pos_tcp[0] - min) / (max - min)
        norm_tcp_pos_y = (self.goal_pos_tcp[1] - min) / (max - min)

        goal_coordinates = (
            int(norm_tcp_pos_x * self.rgb_image_size[0]),
            int(norm_tcp_pos_y * self.rgb_image_size[1]),
        )

        # Draw a circle at the goal
        marker_size = int(self.rgb_image_size[0] / 32)
        overlay_img = cv2.drawMarker(
            tactile_image,
            goal_coordinates,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=marker_size,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        return overlay_img

    def render(self, mode="rgb_array"):
        """
        Most rendering handeled with show_gui, show_tactile flags.
        This is useful for saving videos.
        """

        if mode != "rgb_array":
            return np.array([])

        # get the rgb camera image
        rgb_array = self.get_visual_obs()

        # get the current tactile images and reformat to match rgb array
        tactile_array = self.get_tactile_obs()
        tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)

        # rezise tactile to match rgb if rendering in higher res
        if self._image_size != self.rgb_image_size:
            tactile_array = cv2.resize(tactile_array, tuple(self.rgb_image_size))

        # add goal indicator in approximate position
        tactile_array = self.overlay_goal_on_image(tactile_array)

        # concat the images into a single image
        render_array = np.concatenate([rgb_array, tactile_array], axis=1)

        # setup plot for rendering
        if self._first_render:
            self._first_render = False
            if self._seed is not None:
                self.window_name = "render_window_{}".format(self._seed)
            else:
                self.window_name = "render_window"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # plot rendered image
        if not self._render_closed:
            render_array_rgb = cv2.cvtColor(render_array, cv2.COLOR_BGR2RGB)
            cv2.imshow(self.window_name, render_array_rgb)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(self.window_name)
                self._render_closed = True

        return render_array
