import sys
import gym
import numpy as np
from opensimplex import OpenSimplex

from tactile_gym.robots.arms.robot import Robot
from tactile_gym.assets import add_assets_path
from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv
from tactile_gym.rl_envs.exploration.surface_follow.rest_poses import (
    rest_poses_dict,
)


class BaseSurfaceEnv(BaseTactileEnv):
    def __init__(
        self,
        max_steps=200,
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

        super(BaseSurfaceEnv, self).__init__(max_steps, image_size, show_gui, show_tactile, arm_type=env_modes["arm_type"])

        ## === set modes for easy adjustment ===
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
        self.t_s_core = "no_core"

        # self.goal_dir_for_test = 1
        # self.goal_test_count = 0

        # this well_designed_pos is used for the object and the workframe.
        if self.arm_type in ['mg400', 'magician']:
            self.well_designed_pos = [0.33, 0.0, 0.0]
        else:
            self.well_designed_pos = [0.65, 0.0, 0.0]

        # which t_s to use
        self.t_s_name = env_modes["tactile_sensor_name"]
        if self.noise_mode == "vertical_simplex":
            self.t_s_type = "forward"
        else:
            self.t_s_type = "standard"

        self.t_s_core = "fixed"
        if self.t_s_name == 'tactip':
            self.t_s_dynamics = {"stiffness": 50, "damping": 100, "friction": 10.0}
            self.embed_dist = 0.0025
        elif self.t_s_name == 'digitac':
            self.t_s_dynamics = {'stiffness': 50, 'damping': 100, 'friction': 10.0}
            self.embed_dist = 0.0015
        elif self.t_s_name == 'digit':
            self.t_s_dynamics = {'stiffness': 50, 'damping': 100, 'friction': 10.0}
            self.embed_dist = 0.0015

        # distance from goal to cause termination
        self.termination_dist = 0.01

        # how much penetration of the tip to optimize for

        # setup variables
        self.setup_surface()
        self.setup_action_space()

        # load environment objects
        self.load_environment()
        self.init_surface_and_goal()

        if self.noise_mode == 'vertical_simplex':
            # for vertical surface
            # work frame origin
            # no need to flip the workframe here but need translation offset
            self.workframe_pos = np.array([self.well_designed_pos[0], self.well_designed_pos[1],
                                          0.15+self.height_perturbation_range])
            self.workframe_rpy = np.array([-np.pi, 0.0, 0])

            # limits for tool center point relative to workframe
            TCP_lims = np.zeros(shape=(6, 2))
            TCP_lims[2, 0], TCP_lims[2, 1] = 0, 0  # z lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -self.x_y_extent, self.x_y_extent  # y lims
            TCP_lims[0, 0], TCP_lims[0, 1] = (
                -self.height_perturbation_range,
                +self.height_perturbation_range,
            )  # x lims
            TCP_lims[3, 0], TCP_lims[3, 1] = 0.0, 0.0  # roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] = 0.0, 0.0  # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi / 4, np.pi / 4  # yaw lims

        else:
            # for horizontal surface
            # work frame origin
            self.workframe_pos = np.array(
                [self.well_designed_pos[0], self.well_designed_pos[1], self.height_perturbation_range])
            self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

            # limits for tool center point relative to workframe
            TCP_lims = np.zeros(shape=(6, 2))
            TCP_lims[0, 0], TCP_lims[0, 1] = -self.x_y_extent, +self.x_y_extent  # x lims
            TCP_lims[1, 0], TCP_lims[1, 1] = -self.x_y_extent, +self.x_y_extent  # y lims
            TCP_lims[2, 0], TCP_lims[2, 1] = (
                -self.height_perturbation_range,
                +self.height_perturbation_range,
            )  # z lims
            TCP_lims[3, 0], TCP_lims[3, 1] = -np.pi / 4, +np.pi / 4  # roll lims
            TCP_lims[4, 0], TCP_lims[4, 1] = -np.pi / 4, +np.pi / 4  # pitch lims
            TCP_lims[5, 0], TCP_lims[5, 1] = 0.0, 0.0  # yaw lims

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_name][self.t_s_type]
        # load the robot arm with a t_s attached
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
            t_s_dynamics={"stiffness": 50, "damping": 100, "friction": 10.0},
            show_gui=self._show_gui,
            show_tactile=self._show_tactile,
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
        # setup action space (x,y,z,r,p)
        self.act_dim = self.get_act_dim()

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

            if self.noise_mode == "vertical_simplex":
                # for vertical surface
                max_pos_vel = 0.01  # m/s
                max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

                self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
                self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
                self.z_act_min, self.z_act_max = 0, 0
                self.roll_act_min, self.roll_act_max = 0, 0
                self.pitch_act_min, self.pitch_act_max = 0, 0
                self.yaw_act_min, self.yaw_act_max = -max_ang_vel, max_ang_vel
            else:
                # for horizontal surface
                max_pos_vel = 0.01  # m/s
                max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

                self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
                self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
                self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
                self.roll_act_min, self.roll_act_max = -max_ang_vel, max_ang_vel
                self.pitch_act_min, self.pitch_act_max = -max_ang_vel, max_ang_vel
                self.yaw_act_min, self.yaw_act_max = 0, 0

        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def setup_rgb_obs_camera_params(self):

        if self.arm_type in ["mg400", 'magician']:
            self.rgb_cam_pos = [0.16, 0.0, 0.14]
            self.rgb_cam_dist = 0.45
            self.rgb_cam_yaw = -2
            self.rgb_cam_pitch = -30
            self.rgb_image_size = self._image_size
            # self.rgb_image_size = [512,512]
            self.rgb_fov = 75
            self.rgb_near_val = 0.1
            self.rgb_far_val = 100
        else:
            self.rgb_cam_pos = [0.65, 0.0, 0.05]
            self.rgb_cam_dist = 0.4
            self.rgb_cam_yaw = 90
            self.rgb_cam_pitch = -30
            self.rgb_image_size = self._image_size
            # self.rgb_image_size = [512,512]
            self.rgb_fov = 75
            self.rgb_near_val = 0.1
            self.rgb_far_val = 100

    def setup_surface(self):
        """
        Sets variables for generating a surface from heightfield data.
        """

        # define params for generating coherent random surface using opensimplex
        self.heightfield_grid_scale = 0.006  # mesh grid size
        self.height_perturbation_range = 0.025  # max/min height of surface
        self.num_heightfield_rows, self.num_heightfield_cols = (
            64,
            64,
        )  # n grid points (ensure this matches tactile image dims)
        self.interpolate_noise = 0.05  # "zoom" into the map (opensimplex param)
        self.x_y_extent = 0.15  # limits for x,y TCP coords and goal pos

        # place the surface in the world
        if self.noise_mode == "vertical_simplex":
            self.original_surface_pos = [self.well_designed_pos[0], self.well_designed_pos[1], self.height_perturbation_range]
            self.original_surface_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])
            # get the limits of the surface
            min_x = self.original_surface_pos[0] - (
                (self.num_heightfield_rows / 2) * self.heightfield_grid_scale
            )
            max_x = self.original_surface_pos[0] + (
                (self.num_heightfield_rows / 2) * self.heightfield_grid_scale
            )
            min_y = self.original_surface_pos[1] - (
                (self.num_heightfield_cols / 2) * self.heightfield_grid_scale
            )
            max_y = self.original_surface_pos[1] + (
                (self.num_heightfield_cols / 2) * self.heightfield_grid_scale
            )
            self.interpolate_noise = 0.05
            # no need to plus the height_range on the x coordinate b/c we have rotation matrix.
            self.surface_pos = [self.well_designed_pos[0], self.well_designed_pos[1], 0.15 + self.height_perturbation_range]
            self.surface_orn = self._pb.getQuaternionFromEuler([0.0, -np.pi/2, 0.0])

        else:
            self.surface_pos = [self.well_designed_pos[0], self.well_designed_pos[1], self.height_perturbation_range]
            self.surface_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])
            # get the limits of the surface
            min_x = self.surface_pos[0] - ((self.num_heightfield_rows / 2) * self.heightfield_grid_scale)
            max_x = self.surface_pos[0] + ((self.num_heightfield_rows / 2) * self.heightfield_grid_scale)
            min_y = self.surface_pos[1] - ((self.num_heightfield_cols / 2) * self.heightfield_grid_scale)
            max_y = self.surface_pos[1] + ((self.num_heightfield_cols / 2) * self.heightfield_grid_scale)

        # make a grid of x/y positions for pulling height info from world pos
        self.x_bins = np.linspace(min_x, max_x, self.num_heightfield_rows)
        self.y_bins = np.linspace(min_y, max_y, self.num_heightfield_cols)

    def xy_to_surface_idx(self, x, y):
        """
        input: x,y in world coords
        output: i,j for corresponding nearest heightfield data point.
        """

        # find the idxs corresponding to position
        i = np.digitize(y, self.y_bins)
        j = np.digitize(x, self.x_bins)

        # digitize can return max idx which will throw error if pos is outside of range
        if i == self.num_heightfield_cols:
            i -= 1
        if j == self.num_heightfield_rows:
            j -= 1

        return i, j

    def gen_heigtfield_noisey(self):
        """
        Generates a heightmap using uniform noise which results in sharp differences
        between neighbouring vertices
        """
        heightfield_data = np.zeros(shape=(self.num_heightfield_rows, self.num_heightfield_cols))

        for j in range(int(self.num_heightfield_cols / 2)):
            for i in range(int(self.num_heightfield_rows / 2)):
                height = self.np_random.uniform(0, self.height_perturbation_range * 0.2)
                heightfield_data[2 * i, 2 * j] = height
                heightfield_data[2 * i + 1, 2 * j] = height
                heightfield_data[2 * i, 2 * j + 1] = height
                heightfield_data[2 * i + 1, 2 * j + 1] = height

        return heightfield_data

    def gen_heigtfield_simplex_2d(self):
        """
        Generates a heightmap using OpenSimplex algorithm which results in
        coherent noise across neighbouring vertices.
        Noise in both X and Y directions.
        """

        heightfield_data = np.zeros(shape=(self.num_heightfield_rows, self.num_heightfield_cols))

        for x in range(int(self.num_heightfield_rows)):
            for y in range(int(self.num_heightfield_cols)):

                height = (
                    self.simplex_noise.noise2(x=x * self.interpolate_noise, y=y * self.interpolate_noise)
                    * self.height_perturbation_range
                )
                heightfield_data[x, y] = height

        return heightfield_data

    def gen_heigtfield_simplex_1d(self):
        """
        Generates a heightmap using OpenSimplex algorithm which results in
        coherent noise across neighbouring vertices.
        Noise only in Y direction.
        """

        heightfield_data = np.zeros(shape=(self.num_heightfield_rows, self.num_heightfield_cols))

        for x in range(int(self.num_heightfield_rows)):
            for y in range(int(self.num_heightfield_cols)):

                height = (
                    self.simplex_noise.noise2(x=1 * self.interpolate_noise, y=y * self.interpolate_noise)
                    * self.height_perturbation_range
                )
                heightfield_data[x, y] = height

        return heightfield_data

    def gen_heigtfield_simplex_1d_vertical(self):
        """
        Generates a heightmap using OpenSimplex algorithm which results in
        coherent noise across neighbouring vertices.
        Noise only in Y direction.
        """

        heightfield_data = np.zeros(
            shape=(self.num_heightfield_rows, self.num_heightfield_cols)  # 64,64
        )

        for x in range(int(self.num_heightfield_rows)):
            for y in range(int(self.num_heightfield_cols)):

                height = (
                    self.simplex_noise.noise2(
                        # 0.05, "zoom" into the map (opensimplex param),
                        x=x * self.interpolate_noise, y=1 * self.interpolate_noise
                    )
                    * self.height_perturbation_range  # 0.025  # max/min height of surface
                )
                heightfield_data[x, y] = height

        return heightfield_data

    def init_surface_and_goal(self):
        """
        Loads a surface based on previously set data.
        Also laod a goal indicator that can be moved to new positions on updates.
        """
        # generate heightfield data as zeros, this gets updated to noisey terrain
        self.heightfield_data = np.zeros(shape=(self.num_heightfield_rows, self.num_heightfield_cols))

        self.create_surface()

        # load a goal so that it can have its position updated
        self.goal_indicator = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf"),
            self.surface_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
        )

    def create_surface(self):

        # load surface
        if self.noise_mode == "vertical_simplex":
            self.surface_shape = self._pb.createCollisionShape(
                shapeType=self._pb.GEOM_HEIGHTFIELD,
                meshScale=[self.heightfield_grid_scale, self.heightfield_grid_scale, 1],
                heightfieldTextureScaling=(self.num_heightfield_rows - 1) / 2,
                heightfieldData=self.heightfield_data.flatten(),
                numHeightfieldRows=self.num_heightfield_rows,
                numHeightfieldColumns=self.num_heightfield_cols,
            )
        else:
            self.surface_shape = self._pb.createCollisionShape(
                shapeType=self._pb.GEOM_HEIGHTFIELD,
                meshScale=[self.heightfield_grid_scale, self.heightfield_grid_scale, 1],
                heightfieldTextureScaling=(self.num_heightfield_rows - 1) / 2,
                heightfieldData=self.heightfield_data.flatten(),
                numHeightfieldRows=self.num_heightfield_rows,
                numHeightfieldColumns=self.num_heightfield_cols,
            )

        self.surface_id = self._pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=self.surface_shape)

        self._pb.resetBasePositionAndOrientation(self.surface_id, self.surface_pos, self.surface_orn)

        # change color of surface (keep opacity at 1.0)
        self._pb.changeVisualShape(self.surface_id, -1, rgbaColor=[0, 0.0, 1, 1.0])

        # turn off collisions with surface
        self._pb.setCollisionFilterGroupMask(self.surface_id, -1, 0, 0)

    def update_surface(self):
        """
        Update an already loaded surface with random noise.
        """

        if self.noise_mode == "none":
            self.heightfield_data = np.zeros(shape=(self.num_heightfield_rows, self.num_heightfield_cols))

        elif self.noise_mode == "random":
            self.heightfield_data = self.gen_heigtfield_noisey()

        elif self.noise_mode == "simplex":

            # set seed for simplex noise
            self.simplex_noise = OpenSimplex(seed=self.np_random.randint(1e8))

            if self.movement_mode in ["yz", "yzRx"]:
                self.heightfield_data = self.gen_heigtfield_simplex_1d()

            elif self.movement_mode in ["xyz", "xyzRxRy"]:
                self.heightfield_data = self.gen_heigtfield_simplex_2d()

        elif self.noise_mode == "vertical_simplex":
            # set seed for simplex noise
            self.simplex_noise = OpenSimplex(seed=self.np_random.randint(1e8))
            if self.movement_mode in ["xRz"]:
                self.heightfield_data = self.gen_heigtfield_simplex_1d_vertical()

            else:
                sys.exit("Incorrect movement mode specified")
        else:
            sys.exit("Incorrect noise mode specified")

        # update heightfield
        self._pb.removeBody(self.surface_id)
        self.create_surface()

        # self.surface_shape = self._pb.createCollisionShape(
        #     shapeType=self._pb.GEOM_HEIGHTFIELD,
        #     meshScale=[self.heightfield_grid_scale, self.heightfield_grid_scale, 1],  # unit size
        #     heightfieldTextureScaling=(self.num_heightfield_rows - 1) / 2,  # no need
        #     heightfieldData=self.heightfield_data.flatten(),  # from gen_heigtfield_simplex_1d(), get cordinate and its height
        #     numHeightfieldRows=self.num_heightfield_rows,  # 64
        #     numHeightfieldColumns=self.num_heightfield_cols,  # 64
        #     replaceHeightfieldIndex=self.surface_shape,  # no need
        #     physicsClientId=self._physics_client_id  # no need
        # )

        # create an array for the surface in world coords
        X, Y = np.meshgrid(self.x_bins, self.y_bins)
        self.surface_array = np.dstack((X, Y, self.heightfield_data + self.surface_pos[2]))

        if self.noise_mode == 'vertical_simplex':
            # convert the surface all coordinates into the flip surface (vertical)
            for x_ind in range(len(self.surface_array[0])):
                for y_ind in range(len(self.surface_array[1])):
                    pos = self.surface_array[x_ind][y_ind]
                    pos_surfaceframe, rpy_surfaceframe = self.worldframe_to_surfaceframe(pos, [0, 0, 0])
                    orn_surfaceframe = self._pb.getQuaternionFromEuler(rpy_surfaceframe)
                    flip_pos_surf, flip_orn_surf = self._pb.multiplyTransforms(
                            [0, 0, 0], self.surface_orn, pos_surfaceframe, orn_surfaceframe)
                    flip_rpy_surf = self._pb.getEulerFromQuaternion(flip_orn_surf)

                    flip_pos_worldf, flip_rpy_worldf = self.surfaceframe_to_worldframe(flip_pos_surf, flip_rpy_surf)
                    self.surface_array[x_ind][y_ind] = flip_pos_worldf

        # Create a grid of surface normal vectors for calculating reward
        surface_grad_y, surface_grad_x = np.gradient(self.heightfield_data, self.heightfield_grid_scale)
        self.surface_normals = np.dstack((-surface_grad_x, -surface_grad_y, np.ones_like(self.heightfield_data)))

        # normalise
        n = np.linalg.norm(self.surface_normals, axis=2)
        self.surface_normals[:, :, 0] /= n
        self.surface_normals[:, :, 1] /= n
        self.surface_normals[:, :, 2] /= n

        if self.noise_mode == 'vertical_simplex':
            for x_ind in range(len(self.surface_normals[0])):
                for y_ind in range(len(self.surface_normals[1])):
                    flip_mat = self._pb.getQuaternionFromEuler([0, -np.pi/2, 0])
                    orn_mat = self._pb.getQuaternionFromEuler([0, 0, 0])
                    self.surface_normals[x_ind][y_ind], _ = self._pb.multiplyTransforms(
                            [0, 0, 0], flip_mat, self.surface_normals[x_ind][y_ind], orn_mat)

    def make_goal(self):
        """
        Generate a random position on the current surface.
        Set the directions for automatically moving towards goal.
        """

        self.workframe_directions = [0, 0, 0]

        # generate goal on surface (no variation in x)
        if self.movement_mode in ["yz", "yzRx"]:
            self.workframe_directions[0] = 0
            self.workframe_directions[1] = self.np_random.choice([-1, 1])

        # generate goal on surface (on circumfrence of circle within bounds)
        elif self.movement_mode in ["xyz", "xyzRxRy"]:
            ang = self.np_random.uniform(-np.pi, np.pi)
            self.workframe_directions[0] = np.cos(ang)
            self.workframe_directions[1] = np.sin(ang)

        # generate goal on vertical surface
        elif self.movement_mode in ["xRz"]:
            self.workframe_directions[0] = 0
            self.workframe_directions[1] = self.np_random.choice([-1, 1])
            self.workframe_directions[2] = 0

        # translate from world direction to workframe frame direction
        # in order to auto move towards goal
        self.worldframe_directions = self.robot.arm.workvec_to_worldvec(self.workframe_directions)

        # create the goal in world coords
        if self.noise_mode == 'vertical_simplex':
            # because the goal pos is referred to the x_bins, y_bins, so it needs original surfarce pos
            self.goal_pos_worldframe = [
                self.original_surface_pos[0] + self.x_y_extent * self.worldframe_directions[0],
                self.original_surface_pos[1] + self.x_y_extent * self.worldframe_directions[1],
            ]
            # get z pos from surface
            goal_i, goal_j = self.xy_to_surface_idx(self.goal_pos_worldframe[0], self.goal_pos_worldframe[1])
            self.goal_pos_worldframe = self.surface_array[goal_i, goal_j]
        else:
            self.goal_pos_worldframe = [
                self.surface_pos[0] + self.x_y_extent * self.worldframe_directions[0],
                self.surface_pos[1] + self.x_y_extent * self.worldframe_directions[1],
            ]
            # get z pos from surface
            goal_i, goal_j = self.xy_to_surface_idx(self.goal_pos_worldframe[0], self.goal_pos_worldframe[1])
            self.goal_pos_worldframe.append(self.surface_array[goal_i, goal_j, 2])

        self.goal_rpy_worldframe = [0, 0, 0]
        self.goal_orn_worldframe = self._pb.getQuaternionFromEuler(self.goal_rpy_worldframe)

        # create variables for goal pose in coord frame to use later in easy feature observation
        (
            self.goal_pos_workframe,
            self.goal_rpy_workframe,
        ) = self.robot.arm.worldframe_to_workframe(self.goal_pos_worldframe, self.goal_rpy_worldframe)

        # useful for visualisation, transparent to not interfere with tactile images
        self._pb.resetBasePositionAndOrientation(self.goal_indicator, self.goal_pos_worldframe, self.goal_orn_worldframe)

    def reset_task(self):
        """
        Create new random surface.
        Place goal on new surface.
        """
        # regenerate a new surface
        self.update_surface()

        # define a goal pos on new surface
        self.make_goal()

        # initilise sparse rew at 0
        if self.reward_mode == "sparse":
            self.accum_rew = 0.0

    def update_init_pose(self):
        """
        update the initial pose to be taken on reset, relative to the workframe
        """
        # reset the tcp in the center of the surface at the height of the surface at this point
        center_surf_height = self.heightfield_data[int(self.num_heightfield_rows / 2), int(self.num_heightfield_cols / 2)]

        # need to consist with the flip surface
        if self.noise_mode == 'vertical_simplex':
            init_world_pos = [
                self.surface_pos[0] - (center_surf_height - self.embed_dist),
                self.surface_pos[1],
                self.surface_pos[2],
            ]
        else:
            init_world_pos = [
                self.surface_pos[0],
                self.surface_pos[1],
                self.surface_pos[2] + center_surf_height - self.embed_dist,
            ]
        init_TCP_pos, _ = self.robot.arm.worldframe_to_workframe(init_world_pos, [0, 0, 0])
        init_TCP_rpy = np.array([0.0, 0.0, 0.0])

        return init_TCP_pos, init_TCP_rpy

    def reset(self):
        """
        Reset the environment after an episode terminates.
        """

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevelant when loading/removing larger files
        if self.reset_counter > self.reset_limit:
            self.full_reset()

        # reset vars
        self.reset_counter += 1
        self._env_step_counter = 0

        # update the workframe to a new position relative to surface
        self.reset_task()
        init_TCP_pos, init_TCP_rpy = self.update_init_pose()
        # self.robot.arm.draw_TCP()
        self.robot.reset(reset_TCP_pos=init_TCP_pos, reset_TCP_rpy=init_TCP_rpy)
        # just to change variables to the reset pose incase needed before taking
        # a step
        self.get_step_data()

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def full_reset(self):
        """
        Pybullet can encounter some silent bugs, particularly when unloading and
        reloading objects. This will do a full reset every once in a while to
        clear caches.
        """
        self._pb.resetSimulation()
        self.load_environment()
        self.init_surface_and_goal()
        self.robot.full_reset()
        self.reset_counter = 0

    def encode_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm.
        """
        pass

    def get_step_data(self):

        # get the cur tip pos here for once per step
        (
            self.cur_tcp_pos_worldframe,
            self.cur_tcp_rpy_worldframe,
            self.cur_tcp_orn_worldframe,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_worldframe()
        self.tip_i, self.tip_j = self.xy_to_surface_idx(self.cur_tcp_pos_worldframe[0], self.cur_tcp_pos_worldframe[1])

        done = self.termination()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward()

        return reward, done

    def xyz_dist_to_goal(self):
        """
        xyz L2 distance from the current tip position to the goal.
        """
        dist = np.linalg.norm(np.array(self.cur_tcp_pos_worldframe) - np.array(self.goal_pos_worldframe))
        return dist

    def xy_dist_to_goal(self):
        """
        xy L2 distance from the current tip position to the goal.
        Don't care about height in this case.
        """
        dist = np.linalg.norm(np.array(self.cur_tcp_pos_worldframe[:2]) - np.array(self.goal_pos_worldframe[:2]))
        return dist

    def cos_dist_to_surface_normal(self):
        """
        Distance from current orientation of the TCP to the normal of the nearest
        surface point.
        """

        # get normal vector of nearest surface vertex
        targ_surface_normal = self.surface_normals[self.tip_i, self.tip_j, :]
        # get vector of t_s tip, directed through tip body
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)
        if self.noise_mode == "vertical_simplex":
            init_vector = np.array([-1, 0, 0])
        else:
            init_vector = np.array([0, 0, -1])

        rot_tip_vector = tip_rot_matrix.dot(init_vector)

        # get the cosine similarity/distance between the two vectors
        cos_sim = np.dot(targ_surface_normal, rot_tip_vector) / (
            np.linalg.norm(targ_surface_normal) * np.linalg.norm(rot_tip_vector)
        )
        cos_dist = 1 - cos_sim

        return cos_dist

    def z_dist_to_surface(self):
        """
        L1 dist from current tip height to surface height.
        This could be improved by using raycasting and measuring the distance
        to the surface in the current orientation of the sensor but in practice
        this works just as well with less complexity.
        """
        # get the surface height at the current tip pos
        if self.noise_mode == "vertical_simplex":
            surf_x_pos = self.surface_array[self.tip_i, self.tip_j, 0]
            init_vector = np.array([-self.embed_dist, 0, 0])
        else:
            surf_z_pos = self.surface_array[self.tip_i, self.tip_j, 2]
            init_vector = np.array([0, 0, -self.embed_dist])
        # find the position embedded in the tip based on current orientation
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        rot_tip_vector = tip_rot_matrix.dot(init_vector)
        embedded_tip_pos = self.cur_tcp_pos_worldframe + rot_tip_vector

        # draw to double check
        # self._pb.addUserDebugLine(self.cur_tcp_pos_worldframe, embedded_tip_pos, [1, 0, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        # get the current z position of the tip
        # and calculate the distance between the two
        if self.noise_mode == "vertical_simplex":
            tcp_x_pos = embedded_tip_pos[0]
            dist = np.abs(tcp_x_pos - surf_x_pos)
        else:
            tcp_z_pos = embedded_tip_pos[2]
            dist = np.abs(tcp_z_pos - surf_z_pos)

        return dist

    def termination(self):
        """
        Criteria for terminating an episode.
        """
        # terminate when near to goal
        dist = self.xyz_dist_to_goal()
        if dist < self.termination_dist:
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        """
        pass

    def dense_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        pass

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get the surface height and normal at current tip pos
        targ_surf_height = self.surface_array[self.tip_i, self.tip_j, 2]
        targ_surface_normal = self.surface_normals[self.tip_i, self.tip_j, :]
        targ_surface_normal = self.robot.arm.worldvec_to_workvec(targ_surface_normal)

        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            tcp_orn_workframe,
            tcp_lin_vel_workframe,
            tcp_ang_vel_workframe,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        observation = np.hstack(
            [
                *tcp_pos_workframe,
                *tcp_orn_workframe,
                *tcp_lin_vel_workframe,
                *tcp_ang_vel_workframe,
                *self.goal_pos_workframe,
                targ_surf_height,
                *targ_surface_normal,
            ]
        )
        return observation

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        pass

    """
    Debugging tools
    """

    def draw_tip_normal(self):
        """
        Draws a line in GUI calculated as the normal at the current tip
        orientation
        """
        line_scale = 1.0

        # world pos method
        rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        if self.noise_mode == "vertical_simplex":
            init_vector = np.array([-1, 0, 0]) * line_scale
        else:
            init_vector = np.array([0, 0, -1]) * line_scale

        rot_vector = rot_matrix.dot(init_vector)
        self._pb.addUserDebugLine(
            self.cur_tcp_pos_worldframe,
            self.cur_tcp_pos_worldframe + rot_vector,
            [0, 1, 0],
            parentObjectUniqueId=-1,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )

    def draw_target_normal(self):
        """
        Draws a line in GUI calculated as the normal at the nearest surface
        position.
        """
        line_scale = 1.0

        targ_surface_start_point = self.surface_array[self.tip_i, self.tip_j, :]
        targ_surface_normal = self.surface_normals[self.tip_i, self.tip_j, :] * line_scale

        self._pb.addUserDebugLine(
            targ_surface_start_point,
            targ_surface_start_point + targ_surface_normal,
            [1, 0, 0],
            parentObjectUniqueId=-1,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )

    def plot_surface_normals(self):
        """
        Use to visualise all surface and normal vectors
        """
        X, Y = np.meshgrid(self.x_bins, self.y_bins)

        # find an array of surface normals of height map using numeric method
        surface_grad_y, surface_grad_x = np.gradient(self.heightfield_data, self.heightfield_grid_scale)
        surface_normals = np.dstack((-surface_grad_x, -surface_grad_y, np.ones_like(self.heightfield_data)))

        # normalise
        n = np.linalg.norm(surface_normals, axis=2)
        surface_normals[:, :, 0] /= n
        surface_normals[:, :, 1] /= n
        surface_normals[:, :, 2] /= n

        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.plot_surface(X, Y, self.heightfield_data, cmap="viridis", edgecolor="none")
        ax.quiver(
            X,
            Y,
            self.heightfield_data,
            surface_normals[:, :, 0],
            surface_normals[:, :, 1],
            surface_normals[:, :, 2],
            length=0.0025,
            normalize=False,
        )
        # ax.set_xlim3d(-1,1)
        # ax.set_ylim3d(-1,1)
        # ax.set_zlim3d(-1,1)
        ax.set_title("Surface plot")
        plt.show()
        exit()

    def worldframe_to_surfaceframe(self, pos, rpy):
        """
        Transforms a pose in world frame to a pose in work frame.
        """
        pos = np.array(pos)
        rpy = np.array(rpy)
        orn = np.array(self._pb.getQuaternionFromEuler(rpy))

        inv_surfaceframe_pos, inv_surfaceframe_orn = self._pb.invertTransform(
            self.surface_pos, self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])
        )
        surfaceframe_pos, surfaceframe_orn = self._pb.multiplyTransforms(
            inv_surfaceframe_pos, inv_surfaceframe_orn, pos, orn
        )
        surfaceframe_rpy = self._pb.getEulerFromQuaternion(surfaceframe_orn)

        return np.array(surfaceframe_pos), np.array(surfaceframe_rpy)

    def surfaceframe_to_worldframe(self, pos, rpy):
        """
        Transforms a pose in work frame to a pose in world frame.
        """
        pos = np.array(pos)
        rpy = np.array(rpy)
        orn = np.array(self._pb.getQuaternionFromEuler(rpy))

        worldframe_pos, worldframe_orn = self._pb.multiplyTransforms(
            self.surface_pos, self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), pos, orn
        )
        worldframe_rpy = self._pb.getEulerFromQuaternion(worldframe_orn)

        return np.array(worldframe_pos), np.array(worldframe_rpy)
