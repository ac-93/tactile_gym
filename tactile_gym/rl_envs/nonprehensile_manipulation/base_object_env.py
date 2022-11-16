import numpy as np

from tactile_gym.robots.arms.robot import Robot
from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv


class BaseObjectEnv(BaseTactileEnv):
    def __init__(
        self,
        max_steps=1000,
        image_size=[64, 64],
        env_modes=dict(),
        TCP_lims=np.zeros(0),
        rest_poses=np.zeros(0),
        show_gui=False,
        show_tactile=False,
    ):
        super(BaseObjectEnv, self).__init__(max_steps, image_size, show_gui, show_tactile, arm_type=env_modes["arm_type"])

        # set modes for easy adjustment
        self.movement_mode = env_modes["movement_mode"]
        self.control_mode = env_modes["control_mode"]
        self.observation_mode = env_modes["observation_mode"]
        self.reward_mode = env_modes["reward_mode"]

        # setup variables
        self.setup_object()
        self.setup_action_space()

        # load environment objects
        self.load_environment()
        self.load_object(self.visualise_goal)

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
            t_s_dynamics=self.t_s_dynamics,
            show_gui=self._show_gui,
            show_tactile=self._show_tactile,
        )

    def setup_action_space(self):
        """
        Sets variables used for making network predictions and
        sending correct actions to robot from raw network predictions.
        """
        pass

    def setup_object(self):
        """
        Set vars for loading an object
        """
        pass

    def load_object(self, visualise_goal=True):
        """
        Load an object that is used
        """
        # load temp object and goal indicators so they can be more conveniently updated
        self.obj_id = self._pb.loadURDF(self.object_path, self.init_obj_pos, self.init_obj_orn)

        if visualise_goal:
            self.goal_indicator = self._pb.loadURDF(self.goal_path, self.init_obj_pos, [0, 0, 0, 1], useFixedBase=True)
            self._pb.changeVisualShape(self.goal_indicator, -1, rgbaColor=[1, 0, 0, 0.5])
            self._pb.setCollisionFilterGroupMask(self.goal_indicator, -1, 0, 0)

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        pass

    def make_goal(self):
        """
        Generate a goal pose for the object.
        """
        pass

    def reset_task(self):
        """
        Can be used to reset task specific variables
        """
        pass

    def update_workframe(self):
        """
        Change workframe on reset if needed
        """
        pass

    def update_init_pose(self):
        """
        update the workframe to match object size if varied
        """
        # default doesn't change from workframe origin
        init_TCP_pos = np.array([0.0, 0.0, 0.0])
        init_TCP_rpy = np.array([0.0, 0.0, 0.0])
        return init_TCP_pos, init_TCP_rpy

    def get_obj_pos_worldframe(self):
        """
        Get the current position of the object, return as arrays.
        """
        obj_pos, obj_orn = self._pb.getBasePositionAndOrientation(self.obj_id)
        return np.array(obj_pos), np.array(obj_orn)

    def get_obj_pos_workframe(self):
        obj_pos, obj_orn = self.get_obj_pos_worldframe()
        obj_rpy = self._pb.getEulerFromQuaternion(obj_orn)

        obj_pos_workframe, obj_rpy_workframe = self.robot.arm.worldframe_to_workframe(obj_pos, obj_rpy)
        obj_orn_workframe = self._pb.getQuaternionFromEuler(obj_rpy_workframe)
        return obj_pos_workframe, obj_orn_workframe

    def get_obj_vel_worldframe(self):
        """
        Get the current velocity of the object, return as arrays.
        """
        obj_lin_vel, obj_ang_vel = self._pb.getBaseVelocity(self.obj_id)
        return np.array(obj_lin_vel), np.array(obj_ang_vel)

    def get_obj_vel_workframe(self):
        """
        Get the current velocity of the object, return as arrays.
        """
        obj_lin_vel, obj_ang_vel = self.get_obj_vel_worldframe()
        obj_lin_vel, obj_ang_vel = self.robot.arm.worldvel_to_workvel(obj_lin_vel, obj_ang_vel)
        return np.array(obj_lin_vel), np.array(obj_ang_vel)

    def worldframe_to_objframe(self, pos, rpy):
        """
        Transforms a pose in world frame to a pose in work frame.
        """
        pos = np.array(pos)
        rpy = np.array(rpy)
        orn = np.array(self._pb.getQuaternionFromEuler(rpy))

        inv_objframe_pos, inv_objframe_orn = self._pb.invertTransform(self.cur_obj_pos_worldframe, self.cur_obj_orn_worldframe)
        objframe_pos, objframe_orn = self._pb.multiplyTransforms(inv_objframe_pos, inv_objframe_orn, pos, orn)

        return np.array(objframe_pos), np.array(objframe_orn)

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

        # update the workframe to a new position if randomisations are on
        self.reset_task()
        self.update_workframe()

        init_TCP_pos, init_TCP_rpy = self.update_init_pose()
        self.robot.reset(reset_TCP_pos=init_TCP_pos, reset_TCP_rpy=init_TCP_rpy)

        # for debug
        # set_trace()
        # self.robot.arm.print_joint_pos_vel()

        # reset object
        self.reset_object()

        # define a new goal position based on init pose of object
        self.make_goal()

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
        self.load_object(self.visualise_goal)
        self.robot.full_reset()
        self.reset_counter = 0

    def encode_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to ur5.
        """
        pass

    def xyz_tcp_dist_to_goal(self):
        """
        xyz L2 distance from the current tip position to the goal.
        """
        dist = np.linalg.norm(self.cur_tcp_pos_worldframe - self.goal_pos_worldframe)
        return dist

    def xyz_obj_dist_to_goal(self):
        """
        xyz L2 distance from the current obj position to the goal.
        """
        dist = np.linalg.norm(self.cur_obj_pos_worldframe - self.goal_pos_worldframe)
        return dist

    def xy_obj_dist_to_goal(self):
        """
        xyz L2 distance from the current obj position to the goal.
        """
        dist = np.linalg.norm(self.cur_obj_pos_worldframe[:2] - self.goal_pos_worldframe[:2])
        return dist

    def xyz_tcp_dist_to_obj(self):
        """
        xyz L2 distance from the current tip position to the obj center.
        """
        dist = np.linalg.norm(self.cur_tcp_pos_worldframe - self.cur_obj_pos_worldframe)
        return dist

    def orn_obj_dist_to_goal(self):
        """
        Distance between the current obj orientation and goal orientation.
        """
        dist = np.arccos(np.clip((2 * (np.inner(self.goal_orn_worldframe, self.cur_obj_orn_worldframe) ** 2)) - 1, -1, 1))
        return dist

    def termination(self):
        """
        Criteria for terminating an episode.
        """
        pass

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

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        pass

    """
    Debugging
    """

    def draw_obj_workframe(self):
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0.1, 0, 0],
            [1, 0, 0],
            parentObjectUniqueId=self.obj_id,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0.1, 0],
            [0, 1, 0],
            parentObjectUniqueId=self.obj_id,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 1],
            parentObjectUniqueId=self.obj_id,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )

    def draw_goal_workframe(self):
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0.1, 0, 0],
            [1, 0, 0],
            parentObjectUniqueId=self.goal_id,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0.1, 0],
            [0, 1, 0],
            parentObjectUniqueId=self.goal_id,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 1],
            parentObjectUniqueId=self.goal_id,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )
