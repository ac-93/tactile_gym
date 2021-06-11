import numpy as np

rest_poses_dict = {
    "ur5": {
        "right_angle": np.array(
            [
                0.00,  # world_joint            (fixed)
                -0.21330,  # base_joint         (revolute)
                -2.12767,  # shoulder_joint     (revolute)
                -1.83726,  # elbow_joint        (revolute)
                -0.74633,  # wrist_1_joint      (revolute)
                1.56940,  # wrist_2_joint       (revolute)
                -1.78171,  # wrist_3_joint      (revolute)
                0.00,  # ee_joint               (fixed)
                0.00,  # tactip_ee_joint        (fixed)
                0.00,  # tactip_body_to_adapter (fixed)
                0.00,  # tactip_tip_to_body    (fixed)
                0.00,  # tcp_joint              (fixed)
            ]
        )
    },
    "franka_panda": {
        "right_angle": np.array(
            [
                0.00,  # world_joint         (fixed)
                3.02044,  # panda_joint1     (revolute)
                1.33526,  # panda_joint2     (revolute)
                2.69697,  # panda_joint3     (revolute)
                1.00742,  # panda_joint4     (revolute)
                -2.58092,  # panda_joint5    (revolute)
                2.23643,  # panda_joint6     (revolute)
                6.46838,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_body_to_adapter (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        )
    },
    "kuka_iiwa": {
        "right_angle": np.array(
            [
                0.00,  # world_joint            (fixed)
                -0.54316,  # lbr_iiwa_joint_1   (revolute)
                1.12767,  # lbr_iiwa_joint_2    (revolute)
                3.36399,  # lbr_iiwa_joint_3    (revolute)
                1.44895,  # lbr_iiwa_joint_4    (revolute)
                -3.50353,  # lbr_iiwa_joint_5   (revolute)
                0.60381,  # lbr_iiwa_joint_6    (revolute)
                -0.14553,  # lbr_iiwa_joint_7   (revolute)
                0.00,  # ee_joint               (fixed)
                0.00,  # tactip_ee_joint        (fixed)
                0.00,  # tactip_body_to_adapter (fixed)
                0.00,  # tactip_tip_to_body    (fixed)
                0.00,  # tcp_joint              (fixed)
            ]
        )
    },
}
