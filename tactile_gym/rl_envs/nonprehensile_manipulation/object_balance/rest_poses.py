import numpy as np

rest_poses_dict = {
    "ur5": {
        "standard": np.array(
            [
                0.00,  # world_joint         (fixed)
                0.19826,  # base_joint       (revolute)
                -2.01062,  # shoulder_joint  (revolute)
                -1.96602,  # elbow_joint     (revolute)
                -0.73808,  # wrist_1_joint   (revolute)
                4.71286,  # wrist_2_joint    (revolute)
                -3.34064,  # wrist_3_joint   (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        )
    },
    "franka_panda": {
        "standard": np.array(
            [
                0.00,  # world_joint         (fixed)
                3.87708,  # panda_joint1     (revolute)
                2.13087,  # panda_joint2     (revolute)
                1.40977,  # panda_joint3     (revolute)
                1.12215,  # panda_joint4     (revolute)
                -4.14097,  # panda_joint5    (revolute)
                1.46123,  # panda_joint6     (revolute)
                6.65355,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        )
    },
    "kuka_iiwa": {
        "standard": np.array(
            [
                0.00,  # world_joint          (fixed)
                0.77016,  # lbr_iiwa_joint_1  (revolute)
                1.81201,  # lbr_iiwa_joint_2  (revolute)
                1.86389,  # lbr_iiwa_joint_3  (revolute)
                1.53406,  # lbr_iiwa_joint_4  (revolute)
                -1.33246,  # lbr_iiwa_joint_5 (revolute)
                -1.86156,  # lbr_iiwa_joint_6 (revolute)
                2.40460,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00,  # tcp_joint            (fixed)
            ]
        )
    },
}
