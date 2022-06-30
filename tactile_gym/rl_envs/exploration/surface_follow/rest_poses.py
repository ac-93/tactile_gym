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
                    0.20199342416011004,  # base_joint       (revolute)
                    -1.8581332389746197,  # shoulder_joint  (revolute)
                    -1.8168154715398577,  # elbow_joint     (revolute)
                    -1.0385402849835499,  # wrist_1_joint   (revolute)
                    1.569399439236753,  # wrist_2_joint    (revolute)
                    -1.3656188934713112,  # wrist_3_joint    (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00   # tcp_joint              (fixed)
                ]
            ),

            },
        "digitac":
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
                    0.19148011767408704,  # base_joint       (revolute)
                    -1.92776038604851,  # shoulder_joint  (revolute)
                    -1.7217555613365743,  # elbow_joint     (revolute)
                    -1.0625670745823885,  # wrist_1_joint   (revolute)
                    1.568310282843754,  # wrist_2_joint    (revolute)
                    -1.3737671809549512,  # wrist_3_joint    (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00   # tcp_joint              (fixed)
                ]
            ),
            },
        "digit":
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
                    0.19148011767408704,  # base_joint       (revolute)
                    -1.92776038604851,  # shoulder_joint  (revolute)
                    -1.7217555613365743,  # elbow_joint     (revolute)
                    -1.0625670745823885,  # wrist_1_joint   (revolute)
                    1.568310282843754,  # wrist_2_joint    (revolute)
                    -1.3737671809549512,  # wrist_3_joint    (revolute)
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
    "mg400": {
        "tactip":{
            "forward": np.array(
                [
                    0,                      # j1        (fixed)
                    0.27678229586424996,     # j2_1         (revolute)
                    0.6281543378436832,      # j3_1         (revolute)
                    -0.9033290327498503,     # j4_1          (revolute)
                    0,                      # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # adaptor joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    0.2767807985667566,      # j2_2 = j2_1         (revolute)
                    -0.276782284688448 ,   # j3_2 = -j2_1         (revolute)
                    0.9049031579057567      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            ),
        },
        "digit":{
            "forward": np.array(
                [
                    0,                      # j1        (fixed)
                    0.5905679775553622,     # j2_1         (revolute)
                    0.3143233272531256,     # j3_1         (revolute)
                    -0.904800272812408,     # j4_1          (revolute)
                    0,                      # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    0.5905665736774282,      # j2_2 = j2_1         (revolute)
                    -0.5905665736774282 ,   # j3_2 = -j2_1         (revolute)
                    0.904800272812408      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            ),
        },
        "digitac":{
            "forward": np.array(
                [
                    0,                      # j1        (fixed)
                    0.5212839078833752,     # j2_1         (revolute)
                    0.4422081884778576,  # j3_1         (revolute)
                    -0.9632925252126955,     # j4_1          (revolute)
                    0,                      # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    0.5212821789748887,      # j2_2 = j2_1         (revolute)
                    -0.5212821789748887 ,   # j3_2 = -j2_1         (revolute)
                    0.9632925252126955      # j4_2 = j2_1 + j3_1          (revolute)
                ]
                    ),
        },
        
    }
}
