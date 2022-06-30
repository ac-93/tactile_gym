import numpy as np

from tactile_gym.robots.arms.base_robot_arm import BaseRobotArm


class MG400(BaseRobotArm):
    def __init__(
        self, pb, robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
    ):

        super(MG400, self).__init__(
            pb, robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
        )

        # set info specific to arm
        self.setup_mg400_info()
        self.robot_type = 'MG400'
        # reset the arm to rest poses

        self.reset()

    def setup_mg400_info(self):
        """
        Set some of the parameters used when controlling the mg400
        """
        self.max_force = 1000.0
        self.pos_gain = 1.0
        self.vel_gain = 1.0

        self.num_joints = self._pb.getNumJoints(self.robot_id)

        # joints which can be controlled (not fixed)
        self.control_joint_names = [
            "j1",
            "j2_1",
            "j3_1",
            "j4_1",
            "j5",
            "j2_2",
            "j3_2",
            "j4_2",
        ]

        # pull relevent info for controlling the robot (could pull limits and ranges here if needed)
        self.joint_name_to_index = {}
        self.link_name_to_index = {}
        for i in range(self.num_joints):
            info = self._pb.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self.joint_name_to_index[joint_name] = i
            self.link_name_to_index[link_name] = i

        # get the link and tcp IDs
        self.EE_link_id = self.link_name_to_index["ee_link"]
        self.TCP_link_id = self.link_name_to_index["tcp_link"]

        # get the control and calculate joint ids in list form, useful for pb array methods
        self.control_joint_ids = [
            self.joint_name_to_index[name] for name in self.control_joint_names
        ]
        self.num_control_dofs = len(self.control_joint_ids)

        self.link_4_1_id = self.link_name_to_index["link4_1"]
        self.link_4_2_id = self.link_name_to_index["link4_2"]
        self.link_5_id = self.link_name_to_index["link5"]

        self._pb.setCollisionFilterGroupMask(self.robot_id, self.link_4_1_id, 0, 0)
        self._pb.setCollisionFilterGroupMask(self.robot_id, self.link_4_2_id, 0, 0)
        self._pb.setCollisionFilterGroupMask(self.robot_id, self.link_5_id, 0, 0)
        self._pb.setCollisionFilterGroupMask(self.robot_id, self.TCP_link_id, 0, 0)
        self._pb.setCollisionFilterGroupMask(self.robot_id, self.EE_link_id, 0, 0)
        #
        # self.print_joint_pos_vel()
        # self.draw_TCP()

    def tcp_velocity_control(self, desired_vels):
        """
        Actions specifiy desired velocities in the workframe.
        TCP limits are imposed.

        Jacobian size is irregular so alwys use psuedo inverse
        """
        # check that this won't push the TCP out of limits
        # zero any velocities that will
        capped_desired_vels = self.check_TCP_vel_lims(np.array(desired_vels))

        # convert desired vels from workframe to worldframe
        capped_desired_vels[:3], capped_desired_vels[3:] = self.workvel_to_worldvel(
            capped_desired_vels[:3], capped_desired_vels[3:]
        )

        # get current joint positions and velocities
        q, qd = self.get_current_joint_pos_vel()

        # calculate the jacobian for tcp link
        # used to map joing velocities to TCP velocities
        jac_t, jac_r = self._pb.calculateJacobian(
            self.robot_id,
            self.TCP_link_id,
            [0, 0, 0],
            q,
            qd,
            [0] * self.num_control_dofs,
        )

        # merge into one jacobian matrix
        jac = np.concatenate([np.array(jac_t), np.array(jac_r)])

        # invert the jacobian to map from tcp velocities to joint velocities
        inv_jac = np.linalg.pinv(jac)

        # convert desired velocities from cart space to joint space
        req_joint_vels = np.matmul(inv_jac, capped_desired_vels)
        if self.robot_type == "MG400":
            joint_poses = list(req_joint_vels)
            joint_poses[-3] = joint_poses[1]
            joint_poses[-2] = -joint_poses[1]
            joint_poses[-1] = joint_poses[1] + joint_poses[2]
            req_joint_vels = tuple(joint_poses)
        # apply joint space velocities
        self._pb.setJointMotorControlArray(
            self.robot_id,
            self.control_joint_ids,
            self._pb.VELOCITY_CONTROL,
            targetVelocities=req_joint_vels,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

    def tcp_position_control(self, desired_delta_pose):
        """
        Actions specifiy desired changes in position in the workframe.
        TCP limits are imposed.
        """
        # get current position
        (
            cur_tcp_pos,
            cur_tcp_rpy,
            cur_tcp_orn,
            _,
            _,
        ) = self.get_current_TCP_pos_vel_workframe()

        # add actions to current positions
        target_pos = cur_tcp_pos + np.array(desired_delta_pose[:3])
        target_rpy = cur_tcp_rpy + np.array(desired_delta_pose[3:])

        # limit actions to safe ranges
        target_pos, target_rpy = self.check_TCP_pos_lims(target_pos, target_rpy)

        # convert to worldframe coords for IK
        target_pos, target_rpy = self.workframe_to_worldframe(target_pos, target_rpy)
        target_orn = self._pb.getQuaternionFromEuler(target_rpy)

        # get joint positions using inverse kinematics
        joint_poses = self._pb.calculateInverseKinematics(
            self.robot_id,
            self.TCP_link_id,
            target_pos,
            target_orn,
            restPoses=self.rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-8,
        )

        if self.robot_type == "MG400":
            joint_poses = list(joint_poses)
            joint_poses[-3] = joint_poses[1]
            joint_poses[-2] = -joint_poses[1]
            joint_poses[-1] = joint_poses[1] + joint_poses[2]
            joint_poses = tuple(joint_poses)

        # set joint control
        self._pb.setJointMotorControlArray(
            self.robot_id,
            self.control_joint_ids,
            self._pb.POSITION_CONTROL,
            targetPositions=joint_poses,
            targetVelocities=[0] * self.num_control_dofs,
            positionGains=[self.pos_gain] * self.num_control_dofs,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

        # set target positions for blocking move
        self.target_pos_worldframe = target_pos
        self.target_rpy_worldframe = target_rpy
        self.target_orn_worldframe = target_orn
        self.target_joints = joint_poses

    def tcp_direct_workframe_move(self, target_pos, target_rpy):
        """
        Go directly to a position specified relative to the workframe
        """

        # transform from work_frame to world_frame
        target_pos, target_rpy = self.workframe_to_worldframe(target_pos, target_rpy)
        target_orn = np.array(self._pb.getQuaternionFromEuler(target_rpy))
        # get target joint poses through IK
        joint_poses = self._pb.calculateInverseKinematics(
            self.robot_id,
            self.TCP_link_id,
            target_pos,
            target_orn,
            restPoses=self.rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-8,
        )
        # set joint control
        self._pb.setJointMotorControlArray(
            self.robot_id,
            self.control_joint_ids,
            self._pb.POSITION_CONTROL,
            targetPositions=joint_poses,
            targetVelocities=[0] * self.num_control_dofs,
            positionGains=[self.pos_gain] * self.num_control_dofs,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )
        # set target positions for blocking move
        if self.robot_type == "MG400":
            joint_poses = list(joint_poses)
            joint_poses[-3] = joint_poses[1]
            joint_poses[-2] = -joint_poses[1]
            joint_poses[-1] = joint_poses[1] + joint_poses[2]
            joint_poses = tuple(joint_poses)

        self.target_pos_worldframe = target_pos
        self.target_rpy_worldframe = target_rpy
        self.target_orn_worldframe = target_orn
        self.target_joints = joint_poses
