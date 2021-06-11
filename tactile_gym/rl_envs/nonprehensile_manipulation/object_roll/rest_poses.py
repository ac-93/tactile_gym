import numpy as np

rest_poses_dict = {
    "ur5": {
        "flat": np.array(
            [
                0.00,  # world_joint          (fixed)
                0.16682,  # base_joint        (revolute)
                -2.23156,  # shoulder_joint   (revolute)
                -1.66642,  # elbow_joint      (revolute)
                -0.81399,  # wrist_1_joint    (revolute)
                1.57315,  # wrist_2_joint     (revolute)
                1.74001,  # wrist_3_joint     (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00,  # tcp_joint            (fixed)
            ]
        )
    },
    "franka_panda": {
        "flat": np.array(
            [
                0.00,  # world_joint         (fixed)
                -2.90268,  # panda_joint1    (revolute)
                1.44940,  # panda_joint2     (revolute)
                2.64277,  # panda_joint3     (revolute)
                0.79214,  # panda_joint4     (revolute)
                -2.54438,  # panda_joint5    (revolute)
                2.13612,  # panda_joint6     (revolute)
                -1.74541,  # panda_joint7    (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        )
    },
    "kuka_iiwa": {
        "flat": np.array(
            [
                0.00,  # world_joint          (fixed)
                0.29836,  # lbr_iiwa_joint_1  (revolute)
                1.30348,  # lbr_iiwa_joint_2  (revolute)
                2.60906,  # lbr_iiwa_joint_3  (revolute)
                1.22814,  # lbr_iiwa_joint_4  (revolute)
                -2.38960,  # lbr_iiwa_joint_5 (revolute)
                0.80509,  # lbr_iiwa_joint_6  (revolute)
                2.70994,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00,  # tcp_joint            (fixed)
            ]
        )
    },
}
