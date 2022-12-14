import os
import sys
import numpy as np

from tactile_gym.assets import add_assets_path
from tactile_gym.robots.arms.mg400.mg400 import MG400
from tactile_gym.robots.arms.ur5.ur5 import UR5
from tactile_gym.robots.arms.franka_panda.franka_panda import FrankaPanda
from tactile_gym.robots.arms.kuka_iiwa.kuka_iiwa import KukaIiwa
from tactile_gym.sensors.tactile_sensor import TactileSensor

# clean up printing
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})


class Robot:
    def __init__(
        self,
        pb,
        rest_poses,
        workframe_pos,
        workframe_rpy,
        TCP_lims,
        image_size=[128, 128],
        turn_off_border=False,
        arm_type="ur5",
        t_s_name='tactip',
        t_s_type="standard",
        t_s_core="no_core",
        t_s_dynamics={},
        show_gui=True,
        show_tactile=True,
    ):

        self._pb = pb
        self.arm_type = arm_type
        self.t_s_name = t_s_name
        self.t_s_type = t_s_type
        self.t_s_core = t_s_core

        # load the urdf file
        self.robot_id = self.load_robot()
        if self.arm_type == "ur5":
            self.arm = UR5(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "franka_panda":
            self.arm = FrankaPanda(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "kuka_iiwa":
            self.arm = KukaIiwa(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "mg400":
            self.arm = MG400(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        else:
            sys.exit("Incorrect arm type specified {}".format(self.arm_type))

        # get relevent link ids for turning off collisions, connecting camera, etc
        tactile_link_ids = {}
        tactile_link_ids['body'] = self.arm.link_name_to_index[self.t_s_name+"_body_link"]
        tactile_link_ids['tip'] = self.arm.link_name_to_index[self.t_s_name+"_tip_link"]

        if t_s_type in ["right_angle", 'forward', 'mini_right_angle', 'mini_forward']:
            if self.t_s_name == 'tactip':
                tactile_link_ids['adapter'] = self.arm.link_name_to_index[
                    "tactip_adapter_link"
                ]
            elif self.t_s_name in ['digitac', 'digit']:
                print("TODO: Add the adpater link after get it into the URDF")

        # connect the sensor the tactip
        self.t_s = TactileSensor(
            pb,
            robot_id=self.robot_id,
            tactile_link_ids=tactile_link_ids,
            image_size=image_size,
            turn_off_border=turn_off_border,
            t_s_name=t_s_name,
            t_s_type=t_s_type,
            t_s_core=t_s_core,
            t_s_dynamics=t_s_dynamics,
            show_tactile=show_tactile,
            t_s_num=1
        )

    def load_robot(self):
        """
        Load the robot arm model into pybullet
        """
        self.base_pos = [0, 0, 0]
        self.base_rpy = [0, 0, 0]
        self.base_orn = self._pb.getQuaternionFromEuler(self.base_rpy)
        robot_urdf = add_assets_path(os.path.join(
            "robot_assets",
            self.arm_type,
            self.t_s_name,
            self.arm_type + "_with_" + self.t_s_type + "_" + self.t_s_name + ".urdf",
        ))
        robot_id = self._pb.loadURDF(
            robot_urdf, self.base_pos, self.base_orn, useFixedBase=True
        )

        return robot_id

    def reset(self, reset_TCP_pos, reset_TCP_rpy):
        """
        Reset the pose of the UR5 and t_s
        """
        self.arm.reset()
        self.t_s.reset()

        # move to the initial position
        self.arm.tcp_direct_workframe_move(reset_TCP_pos, reset_TCP_rpy)
        # print("TCP pos wrt work frame:",reset_TCP_pos)
        self.blocking_move(max_steps=1000, constant_vel=0.001)
        # self.arm.print_joint_pos_vel()

    def full_reset(self):
        self.load_robot()
        self.t_s.turn_off_t_s_collisions()

    def step_sim(self):
        """
        Take a step of the simulation whilst applying neccessary forces
        """

        # compensate for the effect of gravity
        # self.arm.draw_TCP() # only works with visuals enabled in urdf file
        self.arm.apply_gravity_compensation()

        # step the simulation
        self._pb.stepSimulation()

        # debugging
        # self.arm.draw_EE()
        # self.arm.draw_TCP() # only works with visuals enabled in urdf file
        # self.arm.draw_workframe()
        # self.arm.draw_TCP_box()
        # self.arm.print_joint_pos_vel()
        # self.arm.print_TCP_pos_vel()
        # self.arm.test_workframe_transforms()
        # self.arm.test_workvec_transforms()
        # self.arm.test_workvel_transforms()
        # self.t_s.draw_camera_frame()
        # self.t_s.draw_t_s_frame()

    def apply_action(
        self,
        motor_commands,
        control_mode="TCP_velocity_control",
        velocity_action_repeat=1,
        max_steps=100,
    ):

        if control_mode == "TCP_position_control":
            self.arm.tcp_position_control(motor_commands)

        elif control_mode == "TCP_velocity_control":
            self.arm.tcp_velocity_control(motor_commands)

        elif control_mode == "joint_velocity_control":
            self.arm.joint_velocity_control(motor_commands)

        else:
            sys.exit("Incorrect control mode specified: {}".format(control_mode))

        if control_mode == "TCP_position_control":
            # repeatedly step the sim until a target pose is met or max iters
            self.blocking_move(max_steps=max_steps, constant_vel=None)

        elif control_mode in ["TCP_velocity_control", "joint_velocity_control"]:
            # apply the action for n steps to match control rate
            for i in range(velocity_action_repeat):
                self.step_sim()
        else:
            # just do one step of the sime
            self.step_sim()

    def blocking_move(
        self,
        max_steps=1000,
        constant_vel=None,
        pos_tol=2e-4,
        orn_tol=1e-3,
        jvel_tol=0.1,
    ):
        """
        step the simulation until a target position has been reached or the max
        number of steps has been reached
        """
        # get target position
        targ_pos = self.arm.target_pos_worldframe
        targ_orn = self.arm.target_orn_worldframe
        targ_j_pos = self.arm.target_joints

        pos_error = 0.0
        orn_error = 0.0
        for i in range(max_steps):

            # get the current position and veloicities (worldframe)
            (
                cur_TCP_pos,
                cur_TCP_rpy,
                cur_TCP_orn,
                _,
                _,
            ) = self.arm.get_current_TCP_pos_vel_worldframe()

            # get the current joint positions and velocities
            cur_j_pos, cur_j_vel = self.arm.get_current_joint_pos_vel()

            # Move with constant velocity (from google-ravens)
            # break large position move to series of small position moves.
            if constant_vel is not None:
                diff_j = np.array(targ_j_pos) - np.array(cur_j_pos)
                norm = np.linalg.norm(diff_j)
                v = diff_j / norm if norm > 0 else np.zeros_like(cur_j_pos)
                step_j = cur_j_pos + v * constant_vel

                # reduce vel if joints are close enough,
                # this helps to acheive final pose
                if all(np.abs(diff_j) < constant_vel):
                    constant_vel /= 2

                # set joint control
                self._pb.setJointMotorControlArray(
                    self.robot_id,
                    self.arm.control_joint_ids,
                    self._pb.POSITION_CONTROL,
                    targetPositions=step_j,
                    targetVelocities=[0.0] * self.arm.num_control_dofs,
                    positionGains=[self.arm.pos_gain] * self.arm.num_control_dofs,
                    velocityGains=[self.arm.vel_gain] * self.arm.num_control_dofs
                )

            # step the simulation
            self.step_sim()

            # calc totoal velocity
            total_j_vel = np.sum(np.abs(cur_j_vel))

            # calculate the difference between target and actual pose
            pos_error = np.sum(np.abs(targ_pos - cur_TCP_pos))
            orn_error = np.arccos(
                np.clip((2 * (np.inner(targ_orn, cur_TCP_orn) ** 2)) - 1, -1, 1)
            )

            # break if the pose error is small enough
            # and the velocity is low enough
            if (pos_error < pos_tol) and (orn_error < orn_tol) and (total_j_vel < jvel_tol):
                break

    def get_tactile_observation(self):
        return self.t_s.get_observation()
