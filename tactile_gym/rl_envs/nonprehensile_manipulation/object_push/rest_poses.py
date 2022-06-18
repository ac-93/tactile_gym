import numpy as np

rest_poses_dict = {
    "ur5": {
        "tactip":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.29446578243858357,  # base_joint         (revolute)
                    -2.1633703222876646,  # shoulder_joint     (revolute)
                    -1.7712875440608364,  # elbow_joint        (revolute)
                    -0.7758826291678864,  # wrist_1_joint      (revolute)
                    1.569501010720629,  # wrist_2_joint       (revolute)
                    -1.8628739133606422,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00,  # tcp_joint              (fixed)
                ]
            )
        },
        "digit":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.2363248329397155,  # base_joint         (revolute)
                    -2.1381281530498035,  # shoulder_joint     (revolute)
                    -1.8208841358171288,  # elbow_joint        (revolute)
                    -0.751838113524854,  # wrist_1_joint      (revolute)
                    1.5711258995033301,  # wrist_2_joint       (revolute)
                    -1.80239847761509,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00   # tcp_joint              (fixed)
                ]
            )
        },
        "digitac":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.24571108391609556,  # base_joint         (revolute)
                    -2.142076416487341,  # shoulder_joint     (revolute)
                    -1.8135315230114846,  # elbow_joint        (revolute)
                    -0.7552488203413393,  # wrist_1_joint      (revolute)
                    1.5711290394202047,  # wrist_2_joint       (revolute)
                    -1.8118003855516092,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00   # tcp_joint              (fixed)
                ]
            )
        },
    },
    "mg400": {
        "tactip":{
            "right_angle": np.array(
                [
                    -0.5059580369524724,     # j1        (revolute)
                    1.2694708511711394,     # j2_1         (revolute)
                    -0.19901995409914455,     # j3_1         (revolute)
                    -1.0721610064154656,     # j4_1          (revolute)
                    0.5045899087172413,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_adaptor_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.269469774233031,      # j2_2 = j2_1         (revolute)
                    -1.269469774233031 ,   # j3_2 = -j2_1         (revolute)
                    1.0704498256453248      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            ),
            "mini_right_angle": np.array(
                [
                    -0.4675810386176251,     # j1        (revolute)
                    1.2330268637269028,     # j2_1         (revolute)
                    -0.042146321181746195,     # j3_1         (revolute)
                    -1.1915354526403177,     # j4_1          (revolute)
                    0.4668115359824357,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_adaptor_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.2330268635741901,      # j2_2 = j2_1         (revolute)
                    -1.2330268635741901 ,   # j3_2 = -j2_1         (revolute)
                    1.1908822248875286      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            )
        },
        "digit":{
            "right_angle": np.array(
                [
                    -0.4558165479388624,     # j1        (revolute)
                    1.2857227247064174,     # j2_1         (revolute)
                    0.26532296230426017,     # j3_1         (revolute)
                    -1.5518769541832729,     # j4_1          (revolute)
                    0.45743009274925944,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.28573249852019,      # j2_2 = j2_1         (revolute)
                    -1.2857285129498681 ,   # j3_2 = -j2_1         (revolute)
                    1.5510764390458196      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            )
        },
        "digitac":{
            "right_angle": np.array(
                [
                    -0.4745979999944637,     # j1        (revolute)
                    1.2836350191938928,     # j2_1         (revolute)
                    0.254159419927845,     # j3_1         (revolute)
                    -1.5395417027560878,     # j4_1          (revolute)
                    0.47634420683617346,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.2838656861791102,      # j2_2 = j2_1         (revolute)
                    -1.283854805915325 ,   # j3_2 = -j2_1         (revolute)
                    1.5380912693333302      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            )
        },
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
