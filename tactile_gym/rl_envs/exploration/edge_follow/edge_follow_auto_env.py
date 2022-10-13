import gym
import numpy as np

from tactile_gym.robots.arms.robot import Robot
from tactile_gym.assets import add_assets_path
from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv
from tactile_gym.rl_envs.exploration.edge_follow.rest_poses import (
    rest_poses_dict,
)
import os
# from ipdb import set_trace
env_modes_default = {
    'movement_mode': 'xy',
    'control_mode': 'TCP_velocity_control',
    'noise_mode': 'fixed_height',
    'observation_mode': 'oracle',
    'reward_mode': 'dense',
    'arm_type': 'mg400',
}


class EdgeFollowAutoEnv(BaseTactileEnv):
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

        super(EdgeFollowAutoEnv, self).__init__(
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
        self.stimulus = env_modes["stimulus"]
        # self.arm_type = "ur5"
        # self.arm_type = "mg400"
        # self.arm_type = 'franka_panda'
        # self.arm_type = 'kuka_iiwa'
        self.stim_id = None

        # which t_s to use
        self.t_s_name = env_modes["tactile_sensor_name"]
        # self.t_s_name = 'tactip'
        # self.t_s_name = 'digit'
        self.t_s_type = "standard"
        # self.t_s_type = "mini_standard"
        self.t_s_core = "no_core"

        # distance from goal to cause termination
        self.termination_dist = 0.005

        # limits
        # this well_designed_pos is used for the object and the workframe.
        TCP_lims = np.zeros(shape=(6, 2))
        if self.arm_type in ['mg400', 'magician']:
            self.well_designed_pos = [0.33, 0.0, 0.0]
            TCP_lims[0, 0], TCP_lims[0, 1] = -0.150, +0.150  # x lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -0.11, +0.11  # y lims
            TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, +0.1  # z lims
            TCP_lims[3, 0], TCP_lims[3, 1] = (-np.pi, np.pi)  if self.movement_mode=="TyTzRxRyRz" else (0 ,0)# roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] = (-np.pi, np.pi)  if self.movement_mode=="TyTzRxRyRz" else (0 ,0) # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi, np.pi  # yaw lims
        else:
            self.well_designed_pos = [0.65, 0.0, 0.0]
            TCP_lims[0, 0], TCP_lims[0, 1] = -0.175, +0.175  # x lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -0.175, +0.175  # y lims
            TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, +0.1  # z lims
            TCP_lims[3, 0], TCP_lims[3, 1] =  (-np.pi, np.pi) if self.movement_mode=="TyTzRxRyRz" else (0 ,0)# roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] =  (-np.pi, np.pi)  if self.movement_mode=="TyTzRxRyRz" else (0 ,0) # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi, np.pi  # yaw lims
        # how much penetration of the tip to optimize for
        # randomly vary this on each episode
        if self.t_s_name == 'tactip':
            self.embed_dist = 0.00
        elif self.t_s_name == 'digit':
            self.embed_dist = 0.0035
        elif self.t_s_name == 'digitac':
            self.embed_dist = 0.0035

        # setup variables
        self.setup_stimulus()
        self.setup_action_space()

        # load environment objects
        self.load_environment()
        self.load_stimulus()

        # work frame origin
        self.workframe_pos = np.array([self.well_designed_pos[0], self.well_designed_pos[1], self.stimulus_height])
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
            self.roll_act_min, self.roll_act_max = (-max_ang_change, max_ang_change)  if self.movement_mode=="TyTzRxRyRz" else (0 ,0)
            self.pitch_act_min, self.pitch_act_max = (-max_ang_change, max_ang_change)  if self.movement_mode=="TyTzRxRyRz" else (0 ,0)
            self.yaw_act_min, self.yaw_act_max = -max_ang_change, max_ang_change

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
            self.roll_act_min, self.roll_act_max = (-max_ang_vel, max_ang_vel)  if self.movement_mode=="TyTzRxRyRz" else (0 ,0)
            self.pitch_act_min, self.pitch_act_max = (-max_ang_vel, max_ang_vel)  if self.movement_mode=="TyTzRxRyRz" else (0 ,0)
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

    def setup_stimulus(self):
        # define an initial position for the objects (world coords)
        # self.edge_pos = [0.65, 0.0, 0.0]
        self.stimulus_pos = self.well_designed_pos
        self.stimulus_height = 0.035
        if self.stimulus == "saddle":
            self.stimulus_pos = [0.65, 0.04, 0.0]
        elif self.stimulus == "saddle_for_3d":
            self.stimulus_pos = [0.65, 0.06, -0.02]

    def load_stimulus(self):
        # load temp edge and goal indicators so they can be more conveniently updated

        if self.stim_id is not None:
            self._pb.removeBody(self.stim_id)
        urdf_name = self.stimulus+".urdf"
        stimulus_path = os.path.join("rl_env_assets/exploration/edge_follow/edge_stimuli/",self.stimulus,urdf_name)

        self.stim_id = self._pb.loadURDF(
            add_assets_path(stimulus_path),
            self.stimulus_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
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

    def update_init_pose(self):
        """
        update the initial pose to be taken on reset, relative to the workframe
        """
        if self.stimulus in ["circle"]:
            self.init_TCP_pos = [0, -0.055, self.embed_dist]
        elif self.stimulus in ["foil"]:
            self.init_TCP_pos = [0, -0.035, self.embed_dist]
        elif self.stimulus in ["square"]:
            self.init_TCP_pos = [0, -0.052, self.embed_dist]
        elif self.stimulus in ["clover"]:
            self.init_TCP_pos = [0, -0.05, self.embed_dist]
        elif self.stimulus in ["saddle_for_3d"]:
            self.init_TCP_pos = [0, -0.06, self.embed_dist +0.005 ]
        elif self.stimulus in ["saddle"]:
            self.init_TCP_pos = [0, -0.04, self.embed_dist -0.015 ]
        self.init_TCP_rpy = np.array([0.0, 0.0, 0.0])

        return self.init_TCP_pos, self.init_TCP_rpy

    def reset(self, stimulus="square"):
        """
        Reset the environment after an episode terminates.
        """
        self.stimulus = stimulus
        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevelant when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()

        self.reset_counter += 1
        self._env_step_counter = 0

        # update the workframe to a new position if embed dist randomisations are on
        self.reset_task()
        self.setup_stimulus()
        self.load_stimulus()
        init_TCP_pos, init_TCP_rpy = self.update_init_pose()
        self.robot.reset(reset_TCP_pos=init_TCP_pos, reset_TCP_rpy=init_TCP_rpy)

        # just to change variables to the reset pose incase needed before taking
        # a step
        self.get_step_data()
        # get the starting observation
        self._observation = self.get_observation()
        # Show the TCP
        # self.robot.arm.draw_TCP()
        self.robot.arm.draw_TCP_long_time()
        return self._observation

    def full_reset(self):
        self._pb.resetSimulation()
        self.load_environment()
        self.load_stimulus()
        self.robot.full_reset()
        self.reset_counter = 0


    def encode_TCP_frame_actions(self, actions):

        encoded_actions = np.zeros(6)

        # get rotation matrix from current tip orientation
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        # define initial vectors
        par_vector = np.array([1, 0, 0])  # outwards from tip
        perp_vector = np.array([0, 1, 0])  # perp to tip

        # find the directions based on initial vectors
        par_tip_direction = tip_rot_matrix.dot(par_vector)
        perp_tip_direction = tip_rot_matrix.dot(perp_vector)

        # transform into workframe frame for sending to robot
        workframe_par_tip_direction = self.robot.arm.worldvec_to_workvec(par_tip_direction)
        workframe_perp_tip_direction = self.robot.arm.worldvec_to_workvec(perp_tip_direction)

        if self.movement_mode == "TyRz":

            # translate the direction
            perp_scale = actions[0]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            if self.stimulus == "foil":
                stimulus_scale_speed = 0.35
            else:
                stimulus_scale_speed = 0.5
            par_scale =  stimulus_scale_speed* self.max_action
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[1]

        elif self.movement_mode == "TyTzRz":

            # translate the direction
            perp_scale = actions[0]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            par_scale =  0.5* self.max_action
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[2] += actions[2]
            encoded_actions[5] += actions[1]


        elif self.movement_mode == "TyTzRxRyRz":

            # translate the direction
            perp_scale = actions[0]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            par_scale = self.max_action
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[2] += actions[2]
            encoded_actions[3] += actions[3]
            encoded_actions[4] += actions[4]
            encoded_actions[5] += actions[1]


        elif self.movement_mode == "TxTyRz":

            # translate the direction
            perp_scale = actions[1]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            par_scale = actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[2]

        # draw a line at these points for debugging
        # extended_par_pos  = self.cur_tcp_pos_worldframe + par_tip_direction
        # extended_perp_pos = self.cur_tcp_pos_worldframe + perp_tip_direction
        # self._pb.addUserDebugLine(self.cur_tcp_pos_worldframe, extended_par_pos, [0, 1, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)
        # self._pb.addUserDebugLine(self.cur_tcp_pos_worldframe, extended_perp_pos, [0, 0, 1], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        return encoded_actions

    def encode_work_frame_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm.
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "xy":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]

        if self.movement_mode == "yRz":
            # encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]
            encoded_actions[5] = actions[1]

        elif self.movement_mode == "xyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            # encoded_actions[2] = -self.max_action
            encoded_actions[5] = actions[2]

        return encoded_actions

    def encode_actions(self, actions):
        # scale and embed actions appropriately
        if self.movement_mode in ["xy", "yRz", "xyRz"]:
            encoded_actions = self.encode_work_frame_actions(actions)
        elif self.movement_mode in ["TyRz","TyTzRz", "TxTyRz", "TyTzRxRyRz"]:
            encoded_actions = self.encode_TCP_frame_actions(actions)
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

        (
            self.cur_tcp_pos_workframe,
            _,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()
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

    def xy_dist_to_origin(self):
        dist = np.linalg.norm(
            np.array(self.cur_tcp_pos_workframe) - np.array(self.init_TCP_pos)
        )
        return dist

    def termination(self):

        # terminate when distance to starting point is < eps after some time steps
        if self._env_step_counter > 250:
            if self.xy_dist_to_origin() < self.termination_dist:
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

        return 0

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
        if self.movement_mode == "yRz":
            return 2
        if self.movement_mode == "TyRz":
            return 2
        if self.movement_mode == "TyTzRz":
            return 3
        if self.movement_mode == "TyTzRxRyRz":
            return 5

            