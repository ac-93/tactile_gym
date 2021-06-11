from tactile_gym.robots.arms.base_robot_arm import BaseRobotArm


class FrankaPanda(BaseRobotArm):
    def __init__(self, pb, robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims):

        super(FrankaPanda, self).__init__(pb, robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims)

        # set info specific to arm
        self.setup_panda_info()

        # reset the arm to rest poses
        self.reset()

    def setup_panda_info(self):
        """
        Set some of the parameters used when controlling the Franka Panda
        """
        self.max_force = 1000.0
        self.pos_gain = 1.0
        self.vel_gain = 1.0

        self.num_joints = self._pb.getNumJoints(self.robot_id)

        # joints which can be controlled (not fixed)
        self.control_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
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

        # get the control and calculate joint ids in list form, useful for pb array methods
        self.control_joint_ids = [self.joint_name_to_index[name] for name in self.control_joint_names]
        self.num_control_dofs = len(self.control_joint_ids)

        # get the link and tcp IDs
        self.EE_link_id = self.link_name_to_index["ee_link"]
        self.TCP_link_id = self.link_name_to_index["tcp_link"]
