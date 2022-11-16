import gym
import numpy as np

from tactile_gym.assets import add_assets_path
from tactile_gym.rl_envs.nonprehensile_manipulation.base_object_env import BaseObjectEnv
from tactile_gym.rl_envs.nonprehensile_manipulation.object_balance.rest_poses import (
    rest_poses_dict,
)
from tactile_gym.utils.pybullet_draw_utils import plot_vector

env_modes_default = {
    "movement_mode": "xy",
    "control_mode": "TCP_velocity_control",
    "object_mode": "pole",
    "rand_gravity": False,
    "rand_embed_dist": False,
    "observation_mode": "oracle",
    "reward_mode": "dense",
}


class ObjectBalanceEnv(BaseObjectEnv):
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
        self._control_rate = 1.0 / 20.0
        self._velocity_action_repeat = int(np.floor(self._control_rate / self._sim_time_step))
        self._max_blocking_pos_move_steps = 10

        # pull params from env_modes specific to push env
        self.object_mode = env_modes["object_mode"]
        self.rand_gravity = env_modes["rand_gravity"]
        self.rand_embed_dist = env_modes["rand_embed_dist"]

        # set which robot arm to use
        self.arm_type = env_modes["arm_type"]
        # self.arm_type = "ur5"
        # self.arm_type = "mg400"
        # self.arm_type = 'franka_panda'
        # self.arm_type = 'kuka_iiwa'

        # which t_s to use
        self.t_s_name = env_modes["tactile_sensor_name"]

        self.t_s_type = "standard"
        self.t_s_core = "no_core"
        self.t_s_dynamics = {"stiffness": 50, "damping": 100, "friction": 10.0}

        # distance from goal to cause termination
        self.termination_dist_deg = 35
        self.termination_dist_pos = 0.1

        # how much penetration of the tip to optimize for
        # randomly vary this on each episode
        if self.t_s_name == "tactip":
            self.embed_dist = 0.0035
        elif self.t_s_name == "digitac":
            self.embed_dist = 0.0015
        elif self.t_s_name == "digit":
            self.embed_dist = 0.0015
        # turn on goal visualisation
        self.visualise_goal = False

        # work frame origin
        self.workframe_pos = np.array([0.55, 0.0, 0.35])
        self.workframe_rpy = np.array([0.0, 0.0, 0.0])

        # limits
        TCP_lims = np.zeros(shape=(6, 2))
        TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, 0.1  # x lims
        TCP_lims[1, 0], TCP_lims[1, 1] = -0.1, 0.1  # y lims
        TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, 0.1  # z lims
        TCP_lims[3, 0], TCP_lims[3, 1] = (
            -45 * np.pi / 180,
            45 * np.pi / 180,
        )  # roll lims
        TCP_lims[4, 0], TCP_lims[4, 1] = (
            -45 * np.pi / 180,
            45 * np.pi / 180,
        )  # pitch lims
        TCP_lims[5, 0], TCP_lims[5, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # yaw lims

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_type]

        # init base env
        super(ObjectBalanceEnv, self).__init__(
            max_steps,
            image_size,
            env_modes,
            TCP_lims,
            rest_poses,
            show_gui,
            show_tactile,
        )

        if self.object_mode == "ball_on_plate":
            self.load_ball()
        if self.object_mode == "spinning_plate":
            self.load_plate_buffer()

        # contrain the plate to the tip for better dynamics
        self.apply_constraints()

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
            self.z_act_min, self.z_act_max = -max_pos_change, max_pos_change
            self.roll_act_min, self.roll_act_max = -max_ang_change, max_ang_change
            self.pitch_act_min, self.pitch_act_max = -max_ang_change, max_ang_change
            self.yaw_act_min, self.yaw_act_max = 0, 0

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
            self.roll_act_min, self.roll_act_max = -max_ang_vel, max_ang_vel
            self.pitch_act_min, self.pitch_act_max = -max_ang_vel, max_ang_vel
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
        self.rgb_cam_pos = [-0.1, 0.0, 0.25]
        self.rgb_cam_dist = 1.0
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -10
        self.rgb_image_size = self._image_size
        # self.rgb_image_size = [512,512]
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100

    def setup_object(self):
        """
        Set vars for loading an object
        """

        if self.object_mode == "pole":
            self.obj_base_width = 0.1
            self.obj_base_height = 0.0025
            self.object_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_balance/pole/pole.urdf")

            # no buffer in this mode
            self.buffer_width = 0.0
            self.buffer_height = 0.0

        elif self.object_mode == "ball_on_plate":
            self.obj_base_width = 0.2
            self.obj_base_height = 0.0025
            self.object_path = add_assets_path(
                "rl_env_assets/nonprehensile_manipulation/object_balance/round_plate/round_plate.urdf"
            )

            # no buffer in this mode
            self.buffer_width = 0.0
            self.buffer_height = 0.0

        elif self.object_mode == "spinning_plate":
            self.obj_base_width = 0.15
            self.obj_base_height = 0.0267

            self.buffer_width = 0.04
            self.buffer_height = 0.026

            if self._show_gui:
                self.object_path = add_assets_path(
                    "rl_env_assets/nonprehensile_manipulation/object_balance/spinning_plate/spinning_plate_tex.urdf"
                )
            else:
                self.object_path = add_assets_path(
                    "rl_env_assets/nonprehensile_manipulation/object_balance/spinning_plate/spinning_plate.urdf"
                )

        # define an initial position for the objects (world coords)
        self.init_obj_pos = [
            self.workframe_pos[0],
            self.workframe_pos[1],
            self.workframe_pos[2] + (self.buffer_height) + (self.obj_base_height / 2) - self.embed_dist,
        ]
        self.init_obj_rpy = np.array([0.0, 0.0, -np.pi / 2])
        self.init_obj_orn = self._pb.getQuaternionFromEuler(self.init_obj_rpy)

    def load_plate_buffer(self):
        plate_buffer_path = add_assets_path(
            "rl_env_assets/nonprehensile_manipulation/object_balance/spinning_plate/plate_buffer.urdf"
        )

        self.init_buffer_pos = [
            self.workframe_pos[0],
            self.workframe_pos[1],
            self.workframe_pos[2] + (self.buffer_height / 2),
        ]
        self.init_buffer_orn = [0, 0, 0, 1]
        self.buffer_id = self._pb.loadURDF(
            plate_buffer_path,
            self.init_buffer_pos,
            self.init_buffer_orn,
            flags=self._pb.URDF_INITIALIZE_SAT_FEATURES,
        )

    def load_ball(self):
        sphere_rad = 0.0025
        scale = 7.5

        if self._show_gui:
            plate_ball_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_balance/sphere/sphere_tex.urdf")
        else:
            plate_ball_path = add_assets_path("rl_env_assets/nonprehensile_manipulation/object_balance/sphere/sphere.urdf")

        self.init_ball_pos = [
            self.workframe_pos[0],
            self.workframe_pos[1],
            self.workframe_pos[2] + sphere_rad * scale,
        ]
        self.init_ball_orn = [0, 0, 0, 1]
        self.ball_id = self._pb.loadURDF(plate_ball_path, self.init_ball_pos, self.init_ball_orn, globalScaling=scale)

        # make sure friction is high so ball rolls not slides
        self._pb.changeDynamics(self.ball_id, -1, lateralFriction=10.0)

    def apply_constraints(self):

        if self.object_mode in ["pole", "ball_on_plate"]:
            obj_to_const_id = self.obj_id
            child_pos = [0, 0, -self.obj_base_height / 2 + self.embed_dist]

        elif self.object_mode == "spinning_plate":
            obj_to_const_id = self.buffer_id
            child_pos = [0, 0, -self.buffer_height / 2 + self.embed_dist]

        self.obj_tip_constraint_id = self._pb.createConstraint(
            self.robot.robot_id,
            self.robot.arm.TCP_link_id,
            obj_to_const_id,
            -1,
            self._pb.JOINT_POINT2POINT,
            # self._pb.JOINT_FIXED,
            jointAxis=[0, 0, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=child_pos,
            parentFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
            childFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
        )

    def update_constraints(self):
        """
        If embed distance changes then update the constraint to match.
        """
        if self.object_mode in ["pole", "ball_on_plate"]:
            self._pb.changeConstraint(
                self.obj_tip_constraint_id,
                jointChildPivot=[0, 0, -self.obj_base_height / 2 + self.embed_dist],
            )

    def reset_task(self):
        """
        Change gravity
        Change embed distance
        """
        # randomise gravity
        if self.rand_gravity:
            gravity = self.np_random.uniform(-1.0, -0.1)
            self._pb.setGravity(0, 0, gravity)
        else:
            self._pb.setGravity(0, 0, -0.1)

        if self.rand_embed_dist:
            if self.t_s_name == "tactip":
                self.embed_dist = self.np_random.uniform(0.003, 0.006)
            elif self.t_s_name == "digitac":
                self.embed_dist = self.np_random.uniform(0.001, 0.0025)
            elif self.t_s_name == "digit":
                self.embed_dist = self.np_random.uniform(0.0015, 0.0025)

            self.init_obj_pos = [
                self.workframe_pos[0],
                self.workframe_pos[1],
                self.workframe_pos[2] + self.obj_base_height / 2 - self.embed_dist,
            ]
            self.update_constraints()

    def reset_plate_buffer(self):
        self._pb.resetBasePositionAndOrientation(self.buffer_id, self.init_buffer_pos, self.init_buffer_orn)

    def reset_ball(self):
        self._pb.resetBasePositionAndOrientation(self.ball_id, self.init_ball_pos, self.init_ball_orn)

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """

        # reset the position of the object
        self._pb.resetBasePositionAndOrientation(self.obj_id, self.init_obj_pos, self.init_obj_orn)

        # make sure linear and angular damping is 0
        num_obj_joints = self._pb.getNumJoints(self.obj_id)
        for link_id in range(-1, num_obj_joints):
            self._pb.changeDynamics(
                self.obj_id,
                link_id,
                linearDamping=0.0,
                angularDamping=0.0,
            )

        # apply random force to objects
        if self.object_mode == "pole":
            self.apply_random_force_base(force_mag=0.1)

        elif self.object_mode == "ball_on_plate":
            self.reset_ball()
            self.apply_random_torque_ball(force_mag=0.001)

        elif self.object_mode == "spinning_plate":
            self.reset_plate_buffer()
            self.apply_random_torque_obj(force_mag=1.0)
            self.apply_random_force_base(force_mag=1.0)

    def apply_random_force_base(self, force_mag):
        """
        Apply a random force to the pole object
        """

        # calculate force
        force_pos = self.init_obj_pos + np.array(
            [
                self.np_random.choice([-1, 1]) * self.np_random.rand() * self.obj_base_width / 2,
                self.np_random.choice([-1, 1]) * self.np_random.rand() * self.obj_base_width / 2,
                0,
            ]
        )

        force_dir = np.array([0, 0, -1])
        force = force_dir * force_mag

        # apply force
        self._pb.applyExternalForce(self.obj_id, -1, force, force_pos, flags=self._pb.WORLD_FRAME)

        # plot force
        plot_vector(force_pos, force_dir)

    def apply_random_torque_obj(self, force_mag):
        """
        Apply a random torque to the plate object
        """
        force_dir = np.array([0, 0, -1])
        force = force_dir * force_mag

        # apply force
        self._pb.applyExternalTorque(self.obj_id, -1, force, flags=self._pb.LINK_FRAME)

    def apply_random_torque_ball(self, force_mag):
        """
        Apply a random torque to the ball object
        """
        force_dir = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1), 0])
        force = force_dir * force_mag

        # apply force
        self._pb.applyExternalTorque(self.ball_id, -1, force, flags=self._pb.LINK_FRAME)

    def full_reset(self):
        """
        Pybullet can encounter some silent bugs, particularly when unloading and
        reloading objects. This will do a full reset every once in a while to
        clear caches.
        """
        self._pb.resetSimulation()
        self.load_environment()
        self.load_object(self.visualise_goal)
        self.robot.full_reset()
        self.apply_constraints()
        self.reset_counter = 0

    def encode_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm.
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "xy":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]

        if self.movement_mode == "xyz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]

        elif self.movement_mode == "RxRy":
            encoded_actions[3] = actions[0]
            encoded_actions[4] = actions[1]

        elif self.movement_mode == "xyRxRy":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[3] = actions[2]
            encoded_actions[4] = actions[3]

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

        # get rl info
        done = self.termination()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward()

        return reward, done

    def check_obj_fall(self):
        """
        Check if the roll and pitch are greater than allowed threshold.
        Check if distance travelled is greater than threshold.
        """

        cur_obj_pos, cur_obj_orn = self.get_obj_pos_worldframe()
        cur_obj_rpy_deg = np.array(self._pb.getEulerFromQuaternion(cur_obj_orn)) * 180 / np.pi
        init_obj_rpy_deg = self.init_obj_rpy * 180 / np.pi

        # calc distance in deg accounting for angle representation
        rpy_dist = np.abs(((cur_obj_rpy_deg - init_obj_rpy_deg) + 180) % 360 - 180)

        # terminate if either roll or pitch off by set distance
        if (rpy_dist[0] > self.termination_dist_deg) or (rpy_dist[1] > self.termination_dist_deg):
            return True

        # moved too far
        pos_dist = np.linalg.norm(cur_obj_pos - self.init_obj_pos)
        if pos_dist > self.termination_dist_pos:
            return True

        return False

    def termination(self):
        """
        Criteria for terminating an episode.
        """

        if self.check_obj_fall():
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        If the object falls the reward is -1.0 else 0.0
        """
        if self.check_obj_fall():
            reward = -1
        else:
            reward = 0.0

        return reward

    def dense_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        reward = 1.0

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
            ]
        )

        return observation

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        if self.movement_mode == "xy":
            return 2
        if self.movement_mode == "xyz":
            return 3
        if self.movement_mode == "RxRy":
            return 2
        if self.movement_mode == "xyRxRy":
            return 4
