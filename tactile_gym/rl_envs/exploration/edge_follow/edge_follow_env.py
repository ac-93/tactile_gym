import gym
import numpy as np

from tactile_gym.robots.arms.robot import Robot
from tactile_gym.assets import add_assets_path
from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv
from tactile_gym.rl_envs.exploration.edge_follow.rest_poses import (
    rest_poses_dict,
)


env_modes_default = {
    'movement_mode': 'xy',
    'control_mode': 'TCP_velocity_control',
    'noise_mode': 'fixed_height',
    'observation_mode': 'oracle',
    'reward_mode': 'dense',
    'arm_type': 'mg400',
}


class EdgeFollowEnv(BaseTactileEnv):
    def __init__(
        self,
        max_steps=250,
        image_size=[64, 64],
        env_modes=env_modes_default,
        show_gui=False,
        show_tactile=False
    ):

        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(
            np.floor(self._control_rate / self._sim_time_step)
        )
        self._max_blocking_pos_move_steps = 10

        super(EdgeFollowEnv, self).__init__(
            max_steps, image_size, show_gui, show_tactile, arm_type=env_modes['arm_type']
        )

        # set modes for easy adjustment
        self.movement_mode = env_modes["movement_mode"]
        self.control_mode = env_modes["control_mode"]
        self.noise_mode = env_modes["noise_mode"]
        self.observation_mode = env_modes["observation_mode"]
        self.reward_mode = env_modes["reward_mode"]

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
        self.t_s_type = "standard"
        # self.t_s_type = "mini_standard"
        self.t_s_core = "no_core"

        # distance from goal to cause termination
        self.termination_dist = 0.01

        # limits
        # this well_designed_pos is used for the object and the workframe.
        TCP_lims = np.zeros(shape=(6, 2))
        if self.arm_type in ['mg400', 'magician']:
            self.well_designed_pos = [0.33, 0.0, 0.0]
            TCP_lims[0, 0], TCP_lims[0, 1] = -0.150, +0.150  # x lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -0.11, +0.11  # y lims
            TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, +0.1  # z lims
            TCP_lims[3, 0], TCP_lims[3, 1] = 0.0, 0.0  # roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] = 0.0, 0.0  # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi, np.pi  # yaw lims
        else:
            self.well_designed_pos = [0.65, 0.0, 0.0]
            TCP_lims[0, 0], TCP_lims[0, 1] = -0.175, +0.175  # x lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -0.175, +0.175  # y lims
            TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, +0.1  # z lims
            TCP_lims[3, 0], TCP_lims[3, 1] = 0.0, 0.0  # roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] = 0.0, 0.0  # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi, np.pi  # yaw lims
        # how much penetration of the tip to optimize for
        # randomly vary this on each episode
        if self.t_s_name == 'tactip':
            self.embed_dist = 0.0035
        elif self.t_s_name == 'digit':
            self.embed_dist = 0.0035
        elif self.t_s_name == 'digitac':
            self.embed_dist = 0.0035

        # setup variables
        self.setup_edge()
        self.setup_action_space()

        # load environment objects
        self.load_environment()
        self.load_edge()

        # work frame origin
        self.workframe_pos = np.array([self.well_designed_pos[0], self.well_designed_pos[1], self.edge_height])
        self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_name][self.t_s_type]

        # load the ur5 with a t_s attached
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
        self.min_action, self.max_action = -0.25, 0.25

        # define action ranges per act dim to rescale output of policy
        if self.control_mode == "TCP_position_control":

            max_pos_change = 0.001  # m per step
            max_ang_change = 1 * (np.pi / 180)  # rad per step

            self.x_act_min, self.x_act_max = -max_pos_change, max_pos_change
            self.y_act_min, self.y_act_max = -max_pos_change, max_pos_change
            self.z_act_min, self.z_act_max = -max_pos_change, max_pos_change
            self.roll_act_min, self.roll_act_max = 0, 0
            self.pitch_act_min, self.pitch_act_max = 0, 0
            self.yaw_act_min, self.yaw_act_max = -max_ang_change, max_ang_change

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
            self.roll_act_min, self.roll_act_max = 0, 0
            self.pitch_act_min, self.pitch_act_max = 0, 0
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
        set the RGB camera position to capture full image of the env task.
        """
        # front view
        if self.arm_type == "mg400":

            self.rgb_cam_pos = [-0.20, -0.0, -0.25]
            self.rgb_cam_dist = 0.85
            self.rgb_cam_roll = 0
        else:
            self.rgb_cam_pos = [0.35, 0.0, -0.25]
            self.rgb_cam_dist = 0.75
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -35
        self.rgb_image_size = self._image_size
        # self.rgb_image_size = [512,512]
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100

        # # side view
        # self.rgb_cam_pos = [0.15, 0.2, 0.05]
        # self.rgb_cam_dist = 0.75
        # self.rgb_cam_yaw = 0
        # self.rgb_cam_pitch = 0
        # self.rgb_image_size = self._image_size
        # # self.rgb_image_size = [512,512]
        # self.rgb_fov = 75
        # self.rgb_near_val = 0.1
        # self.rgb_far_val = 100

    def setup_edge(self):
        # define an initial position for the objects (world coords)
        # self.edge_pos = [0.65, 0.0, 0.0]
        self.edge_pos = self.well_designed_pos
        self.edge_height = 0.035
        if self.arm_type in ['mg400', 'magician']:
            self.edge_len = 0.105
        else:
            self.edge_len = 0.175

    def load_edge(self):
        # load temp edge and goal indicators so they can be more conveniently updated
        if self.arm_type in ['mg400', 'magician']:
            edge_path = "rl_env_assets/exploration/edge_follow/edge_stimuli/long_edge_flat/short_edge.urdf"
        else:
            edge_path = "rl_env_assets/exploration/edge_follow/edge_stimuli/long_edge_flat/long_edge.urdf"
        self.edge_stim_id = self._pb.loadURDF(
            add_assets_path(edge_path),
            self.edge_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
        )
        self.goal_indicator = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf"),
            self.edge_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
        )

    def update_edge(self):

        # load in the edge stimulus
        self.edge_ang = self.np_random.uniform(-np.pi, np.pi)
        self.edge_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, self.edge_ang])
        self._pb.resetBasePositionAndOrientation(
            self.edge_stim_id, self.edge_pos, self.edge_orn
        )

        # place a goal at the end of an edge (world coords)
        self.goal_pos_worldframe = [
            self.edge_pos[0] + (self.edge_len * np.cos(self.edge_ang)),
            self.edge_pos[1] + (self.edge_len * np.sin(self.edge_ang)),
            self.edge_pos[2] + self.edge_height,
        ]
        self.goal_rpy_worldframe = [0, 0, 0]
        self.goal_orn_worldframe = self._pb.getQuaternionFromEuler(
            self.goal_rpy_worldframe
        )

        # create variables for goal pose in workframe to use later in easy feature observation
        (
            self.goal_pos_workframe,
            self.goal_rpy_workframe,
        ) = self.robot.arm.worldframe_to_workframe(
            self.goal_pos_worldframe, self.goal_rpy_worldframe
        )

        self.edge_end_points = np.array(
            [
                [
                    self.edge_pos[0] - (self.edge_len * np.cos(self.edge_ang)),
                    self.edge_pos[1] - (self.edge_len * np.sin(self.edge_ang)),
                    self.edge_pos[2] + self.edge_height,
                ],
                [
                    self.edge_pos[0] + (self.edge_len * np.cos(self.edge_ang)),
                    self.edge_pos[1] + (self.edge_len * np.sin(self.edge_ang)),
                    self.edge_pos[2] + self.edge_height,
                ],
            ]
        )

        # useful for visualisation
        self._pb.resetBasePositionAndOrientation(
            self.goal_indicator, self.goal_pos_worldframe, self.goal_orn_worldframe
        )

    def reset_task(self):
        """
        Randomise amount tip embedded into edge
        Reorientate edge
        """
        # reset the ur5 arm at the origin of the workframe with variation to the embed distance
        if self.noise_mode == "rand_height":
            if self.t_s_name == 'tactip':
                self.embed_dist = self.np_random.uniform(0.0015, 0.0065)
            elif self.t_s_name == 'digit':
                self.embed_dist = self.np_random.uniform(0.0011, 0.0028)
            elif self.t_s_name == 'digitac':
                self.embed_dist = self.np_random.uniform(0.0015, 0.0045)
        # load an edge with random orientation and goal
        self.update_edge()

    def update_init_pose(self):
        """
        update the initial pose to be taken on reset, relative to the workframe
        """
        init_TCP_pos = [0, 0, self.embed_dist]
        # init_TCP_pos = [0,0, 0]
        init_TCP_rpy = np.array([0.0, 0.0, 0.0])

        return init_TCP_pos, init_TCP_rpy

    def reset(self):
        """
        Reset the environment after an episode terminates.
        """

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevelant when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()

        self.reset_counter += 1
        self._env_step_counter = 0

        # update the workframe to a new position if embed dist randomisations are on
        self.reset_task()
        init_TCP_pos, init_TCP_rpy = self.update_init_pose()
        self.robot.reset(reset_TCP_pos=init_TCP_pos, reset_TCP_rpy=init_TCP_rpy)

        # just to change variables to the reset pose incase needed before taking
        # a step
        self.get_step_data()

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def full_reset(self):
        self._pb.resetSimulation()
        self.load_environment()
        self.load_edge()
        self.robot.full_reset()
        self.reset_counter = 0

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to ur5
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "xy":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
        if self.movement_mode == "xyz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
        if self.movement_mode == "xyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[5] = actions[2]
        if self.movement_mode == "xyzRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
            encoded_actions[5] = actions[3]

        return encoded_actions

    def get_step_data(self):

        # get the cur tip pos here for once per step
        (
            self.cur_tcp_pos_worldframe,
            self.cur_tcp_rpy_worldframe,
            self.cur_tcp_orn_worldframe,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_worldframe()

        # get rl info
        done = self.termination()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward()

        return reward, done

    def xy_dist_to_goal(self):
        dist = np.linalg.norm(
            np.array(self.cur_tcp_pos_worldframe[:2])
            - np.array(self.goal_pos_worldframe[:2])
        )
        return dist

    def xyz_dist_to_goal(self):
        dist = np.linalg.norm(
            np.array(self.cur_tcp_pos_worldframe) - np.array(self.goal_pos_worldframe)
        )
        return dist

    def dist_to_center_edge(self):

        # use only x/y dont need z
        p1 = self.edge_end_points[0, :2]
        p2 = self.edge_end_points[1, :2]
        p3 = self.cur_tcp_pos_worldframe[:2]

        # calculate perpendicular distance between EE and edge
        dist = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

        return dist

    def termination(self):

        # terminate when distance to goal is < eps
        if self.xy_dist_to_goal() < self.termination_dist:
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def sparse_reward(self):

        # +1 for reaching goal
        if self.xy_dist_to_goal() < self.termination_dist:
            reward = 1
        else:
            reward = 0

        return reward

    def dense_reward(self):
        W_goal = 1.0
        W_edge = 10.0
        W_yaw = 1.0

        goal_dist = self.xy_dist_to_goal()
        edge_dist = self.dist_to_center_edge()
        yaw_dist = 0

        # sum rewards with multiplicative factors
        reward = -((W_goal * goal_dist) + (W_edge * edge_dist) + (W_yaw * yaw_dist))

        return reward

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get sim info on TCP
        (
            tcp_pos_workframe,
            _,
            _,
            tcp_lin_vel_workframe,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        observation = np.hstack(
            [
                *tcp_pos_workframe,
                *tcp_lin_vel_workframe,
                *self.goal_pos_workframe,
                self.edge_ang,
            ]
        )
        return observation

    def get_act_dim(self):
        if self.movement_mode == "xy":
            return 2
        if self.movement_mode == "xyz":
            return 3
        if self.movement_mode == "xyRz":
            return 3
        if self.movement_mode == "xyzRz":
            return 4
