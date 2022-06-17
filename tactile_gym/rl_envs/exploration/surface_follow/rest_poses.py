import numpy as np

rest_poses_dict = {
    "ur5": {
        "tactip":
            {
            "standard": np.array(
                [
                    0.00,  # world_joint         (fixed)
                    0.16682,  # base_joint       (revolute)
                    -2.18943,  # shoulder_joint  (revolute)
                    -1.65357,  # elbow_joint     (revolute)
                    -0.86897,  # wrist_1_joint   (revolute)
                    1.57315,  # wrist_2_joint    (revolute)
                    1.74001,  # wrist_3_joint    (revolute)
                    0.00,  # ee_joint            (fixed)
                    0.00,  # tactip_ee_joint     (fixed)
                    0.00,  # tactip_tip_to_body (fixed)
                    0.00,  # tcp_joint           (fixed)
                ]
            ),
            "forward": np.array(
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
                    0.00   # tcp_joint              (fixed)
                ]
            ),

            },


    },



    "franka_panda": {
        "standard": np.array(
            [
                0.00,  # world_joint         (fixed)
                3.21338,  # panda_joint1     (revolute)
                1.32723,  # panda_joint2     (revolute)
                2.99814,  # panda_joint3     (revolute)
                0.82782,  # panda_joint4     (revolute)
                -2.97544,  # panda_joint5    (revolute)
                2.14629,  # panda_joint6     (revolute)
                4.65994,  # panda_joint7     (revolute)
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
                0.27082,  # lbr_iiwa_joint_1  (revolute)
                1.22678,  # lbr_iiwa_joint_2  (revolute)
                2.66555,  # lbr_iiwa_joint_3  (revolute)
                1.25536,  # lbr_iiwa_joint_4  (revolute)
                -2.49814,  # lbr_iiwa_joint_5 (revolute)
                0.80933,  # lbr_iiwa_joint_6  (revolute)
                2.76210,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00,  # tcp_joint            (fixed)
            ]
        )
    },
}
