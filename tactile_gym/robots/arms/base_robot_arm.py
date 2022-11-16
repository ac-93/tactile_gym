import numpy as np
import math
import time


class BaseRobotArm:
    def __init__(self, pb, robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims):

        self._pb = pb
        self.rest_poses = rest_poses  # default joint pose for ur5
        self.robot_id = robot_id

        # set up the work frame
        self.set_workframe(workframe_pos, workframe_rpy)
        self.set_TCP_lims(TCP_lims)

    def reset(self):
        """
        Reset the UR5 to its rest positions and hold.
        """
        # reset the joint positions to a rest position
        for i in range(self.num_joints):
            self._pb.resetJointState(self.robot_id, i, self.rest_poses[i])
            self._pb.changeDynamics(self.robot_id, i, linearDamping=0.04, angularDamping=0.04)
            self._pb.changeDynamics(self.robot_id, i, jointDamping=0.01)

        # hold in rest pose
        self._pb.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.control_joint_ids,
            controlMode=self._pb.POSITION_CONTROL,
            targetPositions=self.rest_poses[self.control_joint_ids],
            targetVelocities=[0] * self.num_control_dofs,
            positionGains=[self.pos_gain] * self.num_control_dofs,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=np.zeros(self.num_control_dofs) + self.max_force,
        )

    def set_workframe(self, pos, rpy):
        """
        set the working coordinate frame (should be expressed relative to world frame)
        """
        self.workframe_pos = np.array(pos)
        self.workframe_rpy = np.array(rpy)
        self.workframe_orn = np.array(self._pb.getQuaternionFromEuler(rpy))

    def workframe_to_worldframe(self, pos, rpy):
        """
        Transforms a pose in work frame to a pose in world frame.
        """

        pos = np.array(pos)
        rpy = np.array(rpy)
        orn = np.array(self._pb.getQuaternionFromEuler(rpy))

        worldframe_pos, worldframe_orn = self._pb.multiplyTransforms(self.workframe_pos, self.workframe_orn, pos, orn)
        worldframe_rpy = self._pb.getEulerFromQuaternion(worldframe_orn)

        return np.array(worldframe_pos), np.array(worldframe_rpy)

    def worldframe_to_workframe(self, pos, rpy):
        """
        Transforms a pose in world frame to a pose in work frame.
        """
        pos = np.array(pos)
        rpy = np.array(rpy)
        orn = np.array(self._pb.getQuaternionFromEuler(rpy))

        inv_workframe_pos, inv_workframe_orn = self._pb.invertTransform(self.workframe_pos, self.workframe_orn)
        workframe_pos, workframe_orn = self._pb.multiplyTransforms(inv_workframe_pos, inv_workframe_orn, pos, orn)
        workframe_rpy = self._pb.getEulerFromQuaternion(workframe_orn)

        return np.array(workframe_pos), np.array(workframe_rpy)

    def workvec_to_worldvec(self, workframe_vec):
        """
        Transforms a vector in work frame to a vector in world frame.
        """
        workframe_vec = np.array(workframe_vec)
        rot_matrix = np.array(self._pb.getMatrixFromQuaternion(self.workframe_orn)).reshape(3, 3)
        worldframe_vec = rot_matrix.dot(workframe_vec)

        return np.array(worldframe_vec)

    def worldvec_to_workvec(self, worldframe_vec):
        """
        Transforms a vector in world frame to a vector in work frame.
        """
        worldframe_vec = np.array(worldframe_vec)
        inv_workframe_pos, inv_workframe_orn = self._pb.invertTransform(self.workframe_pos, self.workframe_orn)
        rot_matrix = np.array(self._pb.getMatrixFromQuaternion(inv_workframe_orn)).reshape(3, 3)
        workframe_vec = rot_matrix.dot(worldframe_vec)

        return np.array(workframe_vec)

    def workvel_to_worldvel(self, workframe_pos_vel, workframe_ang_vel):
        """
        Convert linear and angular velocities in workframe to worldframe.
        """
        rot_matrix = np.array(self._pb.getMatrixFromQuaternion(self.workframe_orn)).reshape(3, 3)

        worldframe_pos_vel = rot_matrix.dot(workframe_pos_vel)
        worldframe_ang_vel = rot_matrix.dot(workframe_ang_vel)

        return worldframe_pos_vel, worldframe_ang_vel

    def worldvel_to_workvel(self, worldframe_pos_vel, worldframe_ang_vel):
        """
        Convert linear and angular velocities in worldframe to workframe.
        """

        inv_workframe_pos, inv_workframe_orn = self._pb.invertTransform(self.workframe_pos, self.workframe_orn)
        rot_matrix = np.array(self._pb.getMatrixFromQuaternion(inv_workframe_orn)).reshape(3, 3)

        workframe_pos_vel = rot_matrix.dot(worldframe_pos_vel)
        workframe_ang_vel = rot_matrix.dot(worldframe_ang_vel)

        return workframe_pos_vel, workframe_ang_vel

    def set_TCP_lims(self, lims):
        """
        Used to limit the range of the TCP
        """
        self.TCP_lims = lims

    def get_current_joint_pos_vel(self):
        """
        Get the current joint states of the ur5
        """
        cur_joint_states = self._pb.getJointStates(self.robot_id, self.control_joint_ids)
        cur_joint_pos = [cur_joint_states[i][0] for i in range(self.num_control_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.num_control_dofs)]

        return cur_joint_pos, cur_joint_vel

    def get_current_TCP_pos_vel_worldframe(self):
        """
        Get the current velocity of the TCP
        """
        tcp_state = self._pb.getLinkState(
            self.robot_id,
            self.TCP_link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=False,
        )
        tcp_pos = np.array(tcp_state[0])  # worldLinkPos
        tcp_orn = np.array(tcp_state[1])  # worldLinkOrn
        tcp_rpy = self._pb.getEulerFromQuaternion(tcp_orn)
        tcp_lin_vel = np.array(tcp_state[6])  # worldLinkLinearVelocity
        tcp_ang_vel = np.array(tcp_state[7])  # worldLinkAngularVelocity
        return tcp_pos, tcp_rpy, tcp_orn, tcp_lin_vel, tcp_ang_vel

    def get_current_TCP_pos_vel_workframe(self):
        # get sim info on TCP
        (
            tcp_pos,
            tcp_rpy,
            tcp_orn,
            tcp_lin_vel,
            tcp_ang_vel,
        ) = self.get_current_TCP_pos_vel_worldframe()
        tcp_pos_workframe, tcp_rpy_workframe = self.worldframe_to_workframe(tcp_pos, tcp_rpy)
        tcp_orn_workframe = self._pb.getQuaternionFromEuler(tcp_rpy_workframe)
        tcp_lin_vel_workframe, tcp_ang_vel_workframe = self.worldvel_to_workvel(tcp_lin_vel, tcp_ang_vel)

        return (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            tcp_orn_workframe,
            tcp_lin_vel_workframe,
            tcp_ang_vel_workframe,
        )

    def compute_gravity_compensation(self):
        cur_joint_pos, cur_joint_vel = self.get_current_joint_pos_vel()
        grav_comp_torque = self._pb.calculateInverseDynamics(
            self.robot_id, cur_joint_pos, cur_joint_vel, [0] * self.num_control_dofs
        )
        return np.array(grav_comp_torque)

    def apply_gravity_compensation(self):
        grav_comp_torque = self.compute_gravity_compensation()

        self._pb.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.control_joint_ids,
            controlMode=self._pb.TORQUE_CONTROL,
            forces=grav_comp_torque,
        )

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
        self.target_pos_worldframe = target_pos
        self.target_rpy_worldframe = target_rpy
        self.target_orn_worldframe = target_orn
        self.target_joints = joint_poses

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

    def tcp_velocity_control(self, desired_vels):
        """
        Actions specifiy desired velocities in the workframe.
        TCP limits are imposed.
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
        # be careful of singnularities and non square matrices
        # use pseudo-inverse when this is the case
        # this is all the time for 7 dof arms like panda
        if jac.shape[1] > np.linalg.matrix_rank(jac.T):
            inv_jac = np.linalg.pinv(jac)
        else:
            inv_jac = np.linalg.inv(jac)

        # convert desired velocities from cart space to joint space
        req_joint_vels = np.matmul(inv_jac, capped_desired_vels)

        # apply joint space velocities
        self._pb.setJointMotorControlArray(
            self.robot_id,
            self.control_joint_ids,
            self._pb.VELOCITY_CONTROL,
            targetVelocities=req_joint_vels,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

    def joint_velocity_control(self, desired_joint_vels):
        """
        Actions specify desired joint velicities.
        No Limits are imposed.
        """
        self._pb.setJointMotorControlArray(
            self.robot_id,
            self.control_joint_ids,
            self._pb.VELOCITY_CONTROL,
            targetVelocities=desired_joint_vels,
            positionGains=[self.pos_gain] * self.num_control_dofs,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

    def check_TCP_pos_lims(self, pos, rpy):
        """
        cap the pos at the TCP limits specified
        """
        pos = np.clip(pos, self.TCP_lims[:3, 0], self.TCP_lims[:3, 1])
        rpy = np.clip(rpy, self.TCP_lims[3:, 0], self.TCP_lims[3:, 1])
        return pos, rpy

    def check_TCP_vel_lims(self, vels):
        """
        check whether action will take TCP outside of limits,
        zero any velocities that will.
        """
        cur_tcp_pos, cur_tcp_rpy, _, _, _ = self.get_current_TCP_pos_vel_workframe()

        # get bool arrays for if limits are exceeded and if velocity is in
        # the direction that's exceeded
        exceed_pos_llims = np.logical_and(cur_tcp_pos < self.TCP_lims[:3, 0], vels[:3] < 0)
        exceed_pos_ulims = np.logical_and(cur_tcp_pos > self.TCP_lims[:3, 1], vels[:3] > 0)
        exceed_rpy_llims = np.logical_and(cur_tcp_rpy < self.TCP_lims[3:, 0], vels[3:] < 0)
        exceed_rpy_ulims = np.logical_and(cur_tcp_rpy > self.TCP_lims[3:, 1], vels[3:] > 0)

        # combine all bool arrays into one
        exceeded_pos = np.logical_or(exceed_pos_llims, exceed_pos_ulims)
        exceeded_rpy = np.logical_or(exceed_rpy_llims, exceed_rpy_ulims)
        exceeded = np.concatenate([exceeded_pos, exceeded_rpy])

        # cap the velocities at 0 if limits are exceeded
        capped_vels = np.array(vels)
        capped_vels[np.array(exceeded)] = 0

        return capped_vels

    """
    ==================== Debug Tools ====================
    """

    def print_joint_pos_vel(self):
        joint_pos, joint_vel = self.get_current_joint_pos_vel()

        print("")
        print("joint pos: ", joint_pos)
        print("joint vel: ", joint_vel)

    def print_TCP_pos_vel(self):
        (
            tcp_pos,
            tcpy_rpy,
            tcp_orn,
            tcp_lin_vel,
            tcp_ang_vel,
        ) = self.get_current_TCP_pos_vel_worldframe()
        print("")
        print("tcp pos:     ", tcp_pos)
        print("tcp orn:     ", tcp_orn)
        print("tcp lin vel: ", tcp_lin_vel)
        print("tcp ang vel: ", tcp_ang_vel)

    def draw_EE(self, lifetime=0.1):
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0.1, 0, 0],
            [1, 0, 0],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.EE_link_id,
            lifeTime=lifetime,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0.1, 0],
            [0, 1, 0],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.EE_link_id,
            lifeTime=lifetime,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 1],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.EE_link_id,
            lifeTime=lifetime,
        )

    def draw_TCP(self, lifetime=0.1):
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0.1, 0, 0],
            [1, 0, 0],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.TCP_link_id,
            lifeTime=lifetime,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0.1, 0],
            [0, 1, 0],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.TCP_link_id,
            lifeTime=lifetime,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 1],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.TCP_link_id,
            lifeTime=lifetime,
        )

    def draw_workframe(self, lifetime=0.1):
        rpy = [0, 0, 0]
        self._pb.addUserDebugLine(
            self.workframe_pos,
            self.workframe_to_worldframe([0.1, 0, 0], rpy)[0],
            [1, 0, 0],
            lifeTime=lifetime,
        )
        self._pb.addUserDebugLine(
            self.workframe_pos,
            self.workframe_to_worldframe([0, 0.1, 0], rpy)[0],
            [0, 1, 0],
            lifeTime=lifetime,
        )
        self._pb.addUserDebugLine(
            self.workframe_pos,
            self.workframe_to_worldframe([0, 0, 0.1], rpy)[0],
            [0, 0, 1],
            lifeTime=lifetime,
        )

    def draw_TCP_box(self):
        self.TCP_lims[0, 0], self.TCP_lims[0, 1] = -0.1, +0.1  # x lims
        self.TCP_lims[1, 0], self.TCP_lims[1, 1] = -0.1, +0.1  # y lims
        self.TCP_lims[2, 0], self.TCP_lims[2, 1] = -0.1, +0.1  # z lims

        p1 = [
            self.workframe_pos[0] + self.TCP_lims[0, 0],
            self.workframe_pos[1] + self.TCP_lims[1, 1],
            self.workframe_pos[2],
        ]
        p2 = [
            self.workframe_pos[0] + self.TCP_lims[0, 0],
            self.workframe_pos[1] + self.TCP_lims[1, 0],
            self.workframe_pos[2],
        ]
        p3 = [
            self.workframe_pos[0] + self.TCP_lims[0, 1],
            self.workframe_pos[1] + self.TCP_lims[1, 0],
            self.workframe_pos[2],
        ]
        p4 = [
            self.workframe_pos[0] + self.TCP_lims[0, 1],
            self.workframe_pos[1] + self.TCP_lims[1, 1],
            self.workframe_pos[2],
        ]

        self._pb.addUserDebugLine(p1, p2, [1, 0, 0])
        self._pb.addUserDebugLine(p2, p3, [1, 0, 0])
        self._pb.addUserDebugLine(p3, p4, [1, 0, 0])
        self._pb.addUserDebugLine(p4, p1, [1, 0, 0])

    def test_workframe_transforms(self):
        init_pos = np.array([0, 0, 0])
        init_rpy = np.array([0, 0, 0])
        init_orn = np.array(self._pb.getQuaternionFromEuler(init_rpy))

        workframe_pos, workframe_orn = self.worldframe_to_workframe(init_pos, init_rpy)
        workframe_rpy = np.array(self._pb.getEulerFromQuaternion(workframe_orn))

        worldframe_pos, worldframe_rpy = self.workframe_to_worldframe(workframe_pos, workframe_rpy)
        worldframe_orn = np.array(self._pb.getQuaternionFromEuler(worldframe_rpy))

        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={"float_kind": float_formatter})
        print("")
        print("Init Position:       {}, Init RPY:       {}".format(init_pos, init_rpy))
        print("Workframe Position:  {}, Workframe RPY:  {}".format(workframe_pos, workframe_rpy))
        print("Worldframe Position: {}, Worldframe RPY: {}".format(worldframe_pos, worldframe_rpy))
        print(
            "Equal Pos: {}, Equal RPY: {}".format(
                np.isclose(init_pos, worldframe_pos).all(),
                np.isclose(init_rpy, worldframe_rpy).all(),
            )
        )

    def test_workvec_transforms(self):
        init_vec = np.random.uniform([0, 0, 1])
        work_vec = self.worldvec_to_workvec(init_vec)
        world_vec = self.workvec_to_worldvec(work_vec)

        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={"float_kind": float_formatter})
        print("")
        print("Init Vec:  {}".format(init_vec))
        print("Work Vec:  {}".format(work_vec))
        print("World Vec: {}".format(world_vec))
        print("Equal Vec: {}".format(np.isclose(init_vec, world_vec).all()))

    def test_workvel_transforms(self):
        init_lin_vel = np.random.uniform([0, 0, 1])
        init_ang_vel = np.random.uniform([0, 0, 1])
        work_lin_vel, work_ang_vel = self.worldvel_to_workvel(init_lin_vel, init_ang_vel)
        world_lin_vel, world_ang_vel = self.workvel_to_worldvel(work_lin_vel, work_ang_vel)

        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={"float_kind": float_formatter})
        print("")
        print("Init Lin Vel:  {}, Init Ang Vel:  {}".format(init_lin_vel, init_ang_vel))
        print("Work Lin Vel:  {},  Work Ang Vel: {}".format(work_lin_vel, work_ang_vel))
        print("World Lin Vel: {}, World Ang Vel: {}".format(world_lin_vel, world_ang_vel))
        print(
            "Equal Lin Vel: {}, Equal Ang Vel: {}".format(
                np.isclose(init_lin_vel, world_lin_vel).all(),
                np.isclose(init_ang_vel, world_ang_vel).all(),
            )
        )

    def move_in_circle(self):

        for i in range(360 * 5):
            t = i * np.pi / 180

            rad = 0.15
            pos = [
                self.TCP_pos[0] + rad * math.cos(t),
                self.TCP_pos[1] + rad * math.sin(t),
                self.TCP_pos[2],
            ]

            # convert to world coords for IK
            targ_pos, targ_orn = self.workframe_to_worldframe(pos, [0, 0, 0])

            joint_poses = self._pb.calculateInverseKinematics(
                self.robot_id, self.TCP_link_id, targ_pos, targ_orn, maxNumIterations=5
            )
            joint_poses = joint_poses[: self.num_control_dofs]

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
            self._pb.stepSimulation()
            time.sleep(1.0 / 240)
