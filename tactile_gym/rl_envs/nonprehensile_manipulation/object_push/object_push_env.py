import os
import sys
import gym
import numpy as np
from opensimplex import OpenSimplex

from tactile_gym.assets import add_assets_path
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.rest_poses import (
    rest_poses_dict,
)
from tactile_gym.rl_envs.nonprehensile_manipulation.base_object_env import BaseObjectEnv


env_modes_default = {
    "movement_mode": "yRz",
    "control_mode": "TCP_velocity_control",
    "rand_init_orn": False,
    "rand_obj_mass": False,
    "traj_type": "simplex",
    "observation_mode": "oracle",
    "reward_mode": "dense",
}


class ObjectPushEnv(BaseObjectEnv):
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
        self.rand_init_orn = env_modes["rand_init_orn"]
        self.rand_obj_mass = env_modes["rand_obj_mass"]
        self.traj_type = env_modes["traj_type"]

        # set which robot arm to use
        self.arm_type = env_modes["arm_type"]
        # self.arm_type = "ur5"
        # self.arm_type = "mg400"
        # self.arm_type = 'franka_panda'
        # self.arm_type = 'kuka_iiwa'

        # obj info
        self.obj_width = 0.08
        self.obj_height = 0.08

        # which t_s to use
        self.t_s_name = env_modes["tactile_sensor_name"]
        self.t_s_type = "right_angle"
        self.t_s_core = "fixed"
        if self.t_s_name == 'tactip':
            self.t_s_dynamics = {"stiffness": 50, "damping": 100, "friction": 10.0}
        elif self.t_s_name == 'digitac':
            self.t_s_dynamics = {'stiffness': 300, 'damping': 100, 'friction': 10.0}
        elif self.t_s_name == 'digit':
            self.t_s_dynamics = {'stiffness': 50, 'damping': 200, 'friction': 10.0}
        # distance from goal to cause termination
        self.termination_pos_dist = 0.025

        # turn on goal visualisation
        self.visualise_goal = False

        if self.arm_type in ['mg400', 'magician']:
            # limits
            TCP_lims = np.zeros(shape=(6, 2))
            TCP_lims[0, 0], TCP_lims[0, 1] = -0.0, 0.3  # x lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -0.1, 0.08  # y lims
            TCP_lims[2, 0], TCP_lims[2, 1] = -0.0, 0.0  # z lims
            TCP_lims[3, 0], TCP_lims[3, 1] = -0.0, 0.0  # roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] = -0.0, 0.0  # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # yaw lims

            if self.t_s_name == "tactip":
                # this well_designed_pos is used for the object and the workframe.
                self.well_designed_pos = np.array([0.30, -0.1, self.obj_height/2])
                self.t_s_type = "mini_right_angle"
            else:
                self.well_designed_pos = np.array([0.25, -0.1, self.obj_height/2])
        else:
            # limits
            TCP_lims = np.zeros(shape=(6, 2))
            TCP_lims[0, 0], TCP_lims[0, 1] = -0.0, 0.3  # x lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -0.1, 0.1  # y lims
            TCP_lims[2, 0], TCP_lims[2, 1] = -0.0, 0.0  # z lims
            TCP_lims[3, 0], TCP_lims[3, 1] = -0.0, 0.0  # roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] = -0.0, 0.0  # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # yaw lims

            # this well_designed_pos is used for the object and the workframe.
            self.well_designed_pos = np.array([0.55, -0.20, self.obj_height/2])
        # work frame origin
        # self.workframe_pos = np.array([0.55, -0.15, 0.04])
        self.workframe_pos = self.well_designed_pos
        self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_name][self.t_s_type]

        super(ObjectPushEnv, self).__init__(
            max_steps,
            image_size,
            env_modes,
            TCP_lims,
            rest_poses,
            show_gui,
            show_tactile,
        )
        # lod all the objects for trajectory
        self.load_trajectory()

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
            self.z_act_min, self.z_act_max = -0, 0
            self.roll_act_min, self.roll_act_max = -0, 0
            self.pitch_act_min, self.pitch_act_max = -0, 0
            self.yaw_act_min, self.yaw_act_max = -max_ang_change, max_ang_change

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -0, 0
            self.roll_act_min, self.roll_act_max = -0, 0
            self.pitch_act_min, self.pitch_act_max = -0, 0
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
        self.rgb_cam_pos = [0.1, 0.0, -0.35]
        self.rgb_cam_dist = 1.0
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -45
        self.rgb_image_size = self._image_size
        # self.rgb_image_size = [512,512]
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100

    def setup_object(self):
        """
        Set vars for loading an object
        """
        # currently hardcode these for cube, could pull this from bounding box

        # define an initial position for the objects (world coords)
        self.init_obj_pos = [self.well_designed_pos[0], self.well_designed_pos[1] + self.obj_width / 2, self.obj_height / 2]
        # self.init_obj_pos = [self.well_designed_pos[0], self.well_designed_pos[1] + self.obj_width / 2 +0.06, self.obj_height / 2] # for doraemon
        # self.init_obj_orn = self._pb.getQuaternionFromEuler([-np.pi, 0.0, np.pi / 2])
        self.init_obj_orn = self._pb.getQuaternionFromEuler([-np.pi, 0.0, np.pi / 2])

        # get paths
        self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_push/cube/cube.urdf")
        # self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_push/pink_tea_box/model.urdf")
        # self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_push/doraemon_bowl/model.urdf")
        # self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_push/doraemon_bowl/model.urdf")

        self.goal_path = add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf")

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        # reser the position of the object
        if self.rand_init_orn:
            self.init_obj_ang = self.np_random.uniform(-np.pi / 32, np.pi / 32)
        else:
            self.init_obj_ang = 0.0

        self.init_obj_orn = self._pb.getQuaternionFromEuler([-np.pi, 0.0, np.pi / 2 + self.init_obj_ang])
        self._pb.resetBasePositionAndOrientation(self.obj_id, self.init_obj_pos, self.init_obj_orn)

        # perform object dynamics randomisations
        self._pb.changeDynamics(
            self.obj_id,
            -1,
            lateralFriction=0.065,
            spinningFriction=0.00,
            rollingFriction=0.00,
            restitution=0.0,
            frictionAnchor=1,
            collisionMargin=0.0001,
        )

        if self.rand_obj_mass:
            obj_mass = self.np_random.uniform(0.4, 0.8)
            self._pb.changeDynamics(self.obj_id, -1, mass=obj_mass)

    def load_trajectory(self):

        # relatively easy traj
        self.traj_n_points = 10
        self.traj_spacing = 0.025
        self.traj_max_perturb = 0.1

        # place goals at each point along traj
        self.traj_ids = []
        for i in range(int(self.traj_n_points)):
            pos = [0.0, 0.0, 0.0]
            traj_point_id = self._pb.loadURDF(
                os.path.join(os.path.dirname(__file__), self.goal_path),
                pos,
                [0, 0, 0, 1],
                useFixedBase=True,
            )
            self._pb.changeVisualShape(traj_point_id, -1, rgbaColor=[0, 1, 0, 0.5])
            self._pb.setCollisionFilterGroupMask(traj_point_id, -1, 0, 0)
            self.traj_ids.append(traj_point_id)

    def update_trajectory(self):

        # setup traj arrays
        self.targ_traj_list_id = -1
        self.traj_pos_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_rpy_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_orn_workframe = np.zeros(shape=(self.traj_n_points, 4))

        if self.traj_type == "simplex":
            self.update_trajectory_simplex()
        elif self.traj_type == "straight":
            self.update_trajectory_straight()
        else:
            sys.exit("Incorrect traj_type specified: {}".format(self.traj_type))

        # calc orientation to place object at
        self.traj_rpy_workframe[:, 2] = np.gradient(self.traj_pos_workframe[:, 1], self.traj_spacing)

        for i in range(int(self.traj_n_points)):
            # get workframe orn
            self.traj_orn_workframe[i] = self._pb.getQuaternionFromEuler(self.traj_rpy_workframe[i])

            # convert worldframe
            pos_worldframe, rpy_worldframe = self.robot.arm.workframe_to_worldframe(
                self.traj_pos_workframe[i], self.traj_rpy_workframe[i]
            )
            orn_worldframe = self._pb.getQuaternionFromEuler(rpy_worldframe)

            # place goal
            self._pb.resetBasePositionAndOrientation(self.traj_ids[i], pos_worldframe, orn_worldframe)
            self._pb.changeVisualShape(self.traj_ids[i], -1, rgbaColor=[0, 1, 0, 0.5])

    def update_trajectory_simplex(self):
        """
        Generates smooth trajectory of goals
        """
        # initialise noise
        simplex_noise = OpenSimplex(seed=self.np_random.randint(1e8))
        init_offset = self.obj_width / 2 + self.traj_spacing

        # generate smooth 1d traj using opensimplex
        first_run = True
        for i in range(int(self.traj_n_points)):

            noise = simplex_noise.noise2(x=i * 0.1, y=1) * self.traj_max_perturb

            if first_run:
                init_noise_pos_offset = -noise
                first_run = False

            x = init_offset + (i * self.traj_spacing)
            y = init_noise_pos_offset + noise
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

    def update_trajectory_straight(self):

        # randomly pick traj direction
        traj_ang = self.np_random.uniform(-np.pi / 8, np.pi / 8)
        # traj_ang = 0.0
        init_offset = self.obj_width / 2 + self.traj_spacing

        for i in range(int(self.traj_n_points)):

            dir_x = np.cos(traj_ang)
            dir_y = np.sin(traj_ang)
            dist = i * self.traj_spacing

            x = init_offset + dist * dir_x
            y = dist * dir_y
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

    def make_goal(self):
        """
        Generate a goal place a set distance from the inititial object pose.
        """
        # update the curren trajecory
        self.update_trajectory()

        # set goal as first point along trajectory
        self.update_goal()

    def update_goal(self):
        """
        move goal along trajectory
        """
        # increment targ list
        self.targ_traj_list_id += 1

        if self.targ_traj_list_id >= self.traj_n_points:
            return False
        else:
            self.goal_id = self.traj_ids[self.targ_traj_list_id]

            # get goal pose in world frame
            (
                self.goal_pos_worldframe,
                self.goal_orn_worldframe,
            ) = self._pb.getBasePositionAndOrientation(self.goal_id)
            self.goal_rpy_worldframe = self._pb.getEulerFromQuaternion(self.goal_orn_worldframe)

            # create variables for goal pose in workframe to use later
            self.goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id]
            self.goal_orn_workframe = self.traj_orn_workframe[self.targ_traj_list_id]
            self.goal_rpy_workframe = self.traj_rpy_workframe[self.targ_traj_list_id]

            # change colour of new target goal
            self._pb.changeVisualShape(self.goal_id, -1, rgbaColor=[0, 0, 1, 0.5])

            # change colour of goal just reached
            prev_goal_traj_list_id = self.targ_traj_list_id - 1 if self.targ_traj_list_id > 0 else None
            if prev_goal_traj_list_id is not None:
                self._pb.changeVisualShape(self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0.5])

            return True

    def encode_TCP_frame_actions(self, actions):

        encoded_actions = np.zeros(6)

        # get rotation matrix from current tip orientation
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        # define initial vectors
        par_vector = np.array([1, 0, 0])  # outwards from tip
        perp_vector = np.array([0, -1, 0])  # perp to tip

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
            par_scale = 1.0 * self.max_action
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
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

        if self.movement_mode == "y":
            encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]

        if self.movement_mode == "yRz":
            encoded_actions[0] = self.max_action
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
        if self.movement_mode in ["y", "yRz", "xyRz"]:
            encoded_actions = self.encode_work_frame_actions(actions)
        elif self.movement_mode in ["TyRz", "TxTyRz"]:
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
            self.cur_obj_pos_worldframe,
            self.cur_obj_orn_worldframe,
        ) = self.get_obj_pos_worldframe()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward()

        # get rl info
        done = self.termination()

        return reward, done

    def cos_tcp_dist_to_obj(self):
        """
        Cos distance from current orientation of the TCP to the current
        orientation of the object
        """

        # get normal vector of object
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        obj_init_vector = np.array([1, 0, 0])
        obj_vector = obj_rot_matrix.dot(obj_init_vector)

        # get vector of t_s tip, directed through tip body
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)
        tip_init_vector = np.array([1, 0, 0])
        tip_vector = tip_rot_matrix.dot(tip_init_vector)

        # get the cosine similarity/distance between the two vectors
        cos_sim = np.dot(obj_vector, tip_vector) / (np.linalg.norm(obj_vector) * np.linalg.norm(tip_vector))
        cos_dist = 1 - cos_sim

        ## draw for debugging
        # line_scale = 0.2
        # start_point = self.cur_obj_pos_worldframe
        # normal = obj_vector * line_scale
        # self._pb.addUserDebugLine(start_point, start_point + normal, [0, 1, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)
        #
        # start_point = self.cur_tcp_pos_worldframe
        # normal = tip_vector * line_scale
        # self._pb.addUserDebugLine(start_point, start_point + normal, [1, 0, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        return cos_dist

    def termination(self):
        """
        Criteria for terminating an episode.
        """

        # check if near goal, change the goal if so
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        if obj_goal_pos_dist < self.termination_pos_dist:

            # update the goal (if not at end of traj)
            goal_updated = self.update_goal()

            # if self.targ_traj_list_id > self.traj_n_points-1:
            if not goal_updated:
                return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        +1 is given for each goal reached.
        This is calculated before termination called as that will update the goal.
        """
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        if obj_goal_pos_dist < self.termination_pos_dist:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    def dense_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        obj_goal_orn_dist = self.orn_obj_dist_to_goal()
        tip_obj_orn_dist = self.cos_tcp_dist_to_obj()

        # weights for rewards
        W_obj_goal_pos = 1.0
        W_obj_goal_orn = 1.0
        W_tip_obj_orn = 1.0

        # sum rewards with multiplicative factors
        reward = -(
            (W_obj_goal_pos * obj_goal_pos_dist) + (W_obj_goal_orn * obj_goal_orn_dist) + (W_tip_obj_orn * tip_obj_orn_dist)
        )

        return reward

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get sim info on object
        cur_obj_pos_workframe, cur_obj_orn_workframe = self.get_obj_pos_workframe()
        cur_obj_rpy_workframe = self._pb.getEulerFromQuaternion(cur_obj_orn_workframe)
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
                *tcp_rpy_workframe,
                *tcp_lin_vel_workframe,
                *tcp_ang_vel_workframe,
                *cur_obj_pos_workframe,
                *cur_obj_rpy_workframe,
                *cur_obj_lin_vel_workframe,
                *cur_obj_ang_vel_workframe,
                *self.goal_pos_workframe,
                *self.goal_rpy_workframe,
            ]
        )

        return observation

    def get_extended_feature_array(self):
        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        feature_array = np.array(
            [
                *tcp_pos_workframe,
                *tcp_rpy_workframe,
                *self.goal_pos_workframe,
                *self.goal_rpy_workframe,
            ]
        )
        return feature_array

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        if self.movement_mode == "y":
            return 1
        if self.movement_mode == "yRz":
            return 2
        if self.movement_mode == "xyRz":
            return 3
        if self.movement_mode == "TyRz":
            return 2
        if self.movement_mode == "TxTyRz":
            return 3
