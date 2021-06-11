import numpy as np

rest_poses_dict = {
    "ur5": {
        "standard": np.array(
            [
                0.00,  # world_joint         (fixed)
                0.16869,  # base_joint       (revolute)
                -2.16671,  # shoulder_joint  (revolute)
                -1.64546,  # elbow_joint     (revolute)
                -0.89850,  # wrist_1_joint   (revolute)
                1.57131,  # wrist_2_joint    (revolute)
                1.74028,  # wrist_3_joint    (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        ),
        "flat": np.array(
            [
                0.00,  # world_joint         (fixed)
                0.16869,  # base_joint       (revolute)
                -2.16671,  # shoulder_joint  (revolute)
                -1.64546,  # elbow_joint     (revolute)
                -0.89850,  # wrist_1_joint   (revolute)
                1.57131,  # wrist_2_joint    (revolute)
                1.74028,  # wrist_3_joint    (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        ),
        "right_angle": np.array(
            [
                0.00,  # world_joint            (fixed)
                -0.21354,  # base_joint         (revolute)
                -2.12786,  # shoulder_joint     (revolute)
                -1.83677,  # elbow_joint        (revolute)
                -0.74768,  # wrist_1_joint      (revolute)
                1.57052,  # wrist_2_joint       (revolute)
                -1.7821,  # wrist_3_joint       (revolute)
                0.00,  # ee_joint               (fixed)
                0.00,  # tactip_ee_joint        (fixed)
                0.00,  # tactip_body_to_adapter (fixed)
                0.00,  # tactip_tip_to_body    (fixed)
                0.00,  # tcp_joint              (fixed)
            ]
        ),
    },
    "franka_panda": {
        "standard": np.array(
            [
                0.00,  # world_joint         (fixed)
                3.21694,  # panda_joint1     (revolute)
                1.25927,  # panda_joint2     (revolute)
                2.99256,  # panda_joint3     (revolute)
                0.85668,  # panda_joint4     (revolute)
                -2.9800,  # panda_joint5     (revolute)
                2.10365,  # panda_joint6     (revolute)
                4.66007,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        ),
        "flat": np.array(
            [
                0.00,  # world_joint         (fixed)
                3.21694,  # panda_joint1     (revolute)
                1.25927,  # panda_joint2     (revolute)
                2.99256,  # panda_joint3     (revolute)
                0.85668,  # panda_joint4     (revolute)
                -2.9800,  # panda_joint5     (revolute)
                2.10365,  # panda_joint6     (revolute)
                4.66007,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        ),
        "right_angle": np.array(
            [
                0.00,  # world_joint         (fixed)
                3.22528,  # panda_joint1     (revolute)
                1.41795,  # panda_joint2     (revolute)
                2.69266,  # panda_joint3     (revolute)
                0.76964,  # panda_joint4     (revolute)
                -2.61941,  # panda_joint5    (revolute)
                2.10635,  # panda_joint6     (revolute)
                6.79505,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_adapter_joint(fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        ),
    },
    "kuka_iiwa": {
        "standard": np.array(
            [
                0.00,  # world_joint          (fixed)
                0.28225,  # lbr_iiwa_joint_1  (revolute)
                1.17260,  # lbr_iiwa_joint_2  (revolute)
                2.6488,  # lbr_iiwa_joint_3   (revolute)
                1.28030,  # lbr_iiwa_joint_4  (revolute)
                -2.51225,  # lbr_iiwa_joint_5 (revolute)
                0.84163,  # lbr_iiwa_joint_6  (revolute)
                2.76630,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00,  # tcp_joint            (fixed)
            ]
        ),
        "flat": np.array(
            [
                0.00,  # world_joint          (fixed)
                0.276547,  # lbr_iiwa_joint_1 (revolute)
                1.19931,  # lbr_iiwa_joint_2  (revolute)
                2.657019,  # lbr_iiwa_joint_3 (revolute)
                1.26779,  # lbr_iiwa_joint_4  (revolute)
                -2.50528,  # lbr_iiwa_joint_5 (revolute)
                0.82604,  # lbr_iiwa_joint_6  (revolute)
                2.76440,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00,  # tcp_joint            (fixed)
            ]
        ),
        "right_angle": np.array(
            [
                0.00,  # world_joint           (fixed)
                -0.22299,  # lbr_iiwa_joint_1  (revolute)
                1.18910,  # lbr_iiwa_joint_2   (revolute)
                3.31743,  # lbr_iiwa_joint_3   (revolute)
                1.232663,  # lbr_iiwa_joint_4  (revolute)
                -3.388320,  # lbr_iiwa_joint_5 (revolute)
                0.739315,  # lbr_iiwa_joint_6  (revolute)
                0.026109,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint              (fixed)
                0.00,  # tactip_ee_joint       (fixed)
                0.00,  # tactip_adapter_joint  (fixed)
                0.00,  # tactip_tip_to_body   (fixed)
                0.00,  # tcp_joint             (fixed)
            ]
        ),
    },
}
