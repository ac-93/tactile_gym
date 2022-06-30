import numpy as np

rest_poses_dict = {
    "ur5": {
        "tactip": {
            "standard": np.array(
                [
                    0.00,  # world_joint         (fixed)
                    0.166827,  # base_joint       (revolute)
                    -2.16515,  # shoulder_joint  (revolute)
                    -1.64365,  # elbow_joint     (revolute)
                    -0.90317,  # wrist_1_joint   (revolute)
                    1.57315,  # wrist_2_joint    (revolute)
                    1.74001,  # wrist_3_joint    (revolute)
                    0.00,  # ee_joint            (fixed)
                    0.00,  # tactip_ee_joint     (fixed)
                    0.00,  # tactip_tip_to_body (fixed)
                    0.00   # tcp_joint           (fixed)
                ]
            ),

        },
        "digit": {
            "standard": np.array(
                [
                    0.00,  # world_joint         (fixed)
                    0.1666452116249431,  # base_joint       (revolute)
                    -2.2334888481855204,  # shoulder_joint  (revolute)
                    -1.6642245054428424,  # elbow_joint     (revolute)
                    -0.8142762445463524,  # wrist_1_joint   (revolute)
                    1.573151527964482,  # wrist_2_joint    (revolute)
                    1.7398309441833082,  # wrist_3_joint    (revolute)
                    0.00,  # ee_joint            (fixed)
                    0.00,  # tactip_ee_joint     (fixed)
                    0.00,  # tactip_tip_to_body (fixed)
                    0.00   # tcp_joint           (fixed)
                ]
            ),
        },
        "digitac": {
            "standard": np.array(
                [
                    0.00,  # world_joint         (fixed)
                    0.16664443404149898,  # base_joint       (revolute)
                    -2.2242489977536737,  # shoulder_joint  (revolute)
                    -1.6618744232210114,  # elbow_joint     (revolute)
                    -0.8258663681806591,  # wrist_1_joint   (revolute)
                    1.5731514988184077,  # wrist_2_joint    (revolute)
                    1.7398302172182332,  # wrist_3_joint    (revolute)
                    0.00,  # ee_joint            (fixed)
                    0.00,  # tactip_ee_joint     (fixed)
                    0.00,  # tactip_tip_to_body (fixed)
                    0.00   # tcp_joint           (fixed)
                ]
            ),
        },

        "franka_panda": {
            "standard": np.array(
                [
                    0.00,  # world_joint         (fixed)
                    3.21456,  # panda_joint1     (revolute)
                    1.30233,  # panda_joint2     (revolute)
                    2.99673,  # panda_joint3     (revolute)
                    0.83832,  # panda_joint4     (revolute)
                    -2.97647,  # panda_joint5    (revolute)
                    2.13176,  # panda_joint6     (revolute)
                    4.65986,  # panda_joint7     (revolute)
                    0.00,  # ee_joint            (fixed)
                    0.00,  # tactip_ee_joint     (fixed)
                    0.00,  # tactip_tip_to_body (fixed)
                    0.00   # tcp_joint           (fixed)
                ]
            )
        }
    },
    "kuka_iiwa": {
        "standard": np.array(
            [
                0.00,  # world_joint          (fixed)
                0.27440,  # lbr_iiwa_joint_1  (revolute)
                1.20953,  # lbr_iiwa_joint_2  (revolute)
                2.66025,  # lbr_iiwa_joint_3  (revolute)
                1.26333,  # lbr_iiwa_joint_4  (revolute)
                -2.50256,  # lbr_iiwa_joint_5 (revolute)
                0.81968,  # lbr_iiwa_joint_6  (revolute)
                2.76347,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00   # tcp_joint            (fixed)
            ]
        )
    },
    "mg400": {
        "tactip": {
            "standard": np.array(
                [
                    0,                      # j1        (fixed)
                    1.1199979523765513,     # j2_1         (revolute)
                    -0.027746434948259045,  # j3_1         (revolute)
                    -1.094390587897371,     # j4_1          (revolute)
                    0.000795099112695166,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.120002713232204,      # j2_2 = j2_1         (revolute)
                    -1.1199729024887553,   # j3_2 = -j2_1         (revolute)
                    1.0922685386653785      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            ),
        },
        "digit": {
            "standard": np.array(
                [
                    0,                      # j1        (fixed)
                    1.3190166816731614,     # j2_1         (revolute)
                    -0.057932730559221525,  # j3_1         (revolute)
                    -1.2611243932983605,     # j4_1          (revolute)
                    0.0006084288058448784,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.3190195840338783,      # j2_2 = j2_1         (revolute)
                    -1.3189925313906967,   # j3_2 = -j2_1         (revolute)
                    1.2610906509351185      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            ),
        },

        "digitac": {
            "standard": np.array(
                [
                    0,                      # j1        (fixed)
                    1.3223687315585777,     # j2_1         (revolute)
                    -0.06290495221125363,  # j3_1         (revolute)
                    -1.2594762221064615,     # j4_1          (revolute)
                    0.0006084288058448784,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.3223720640647498,      # j2_2 = j2_1         (revolute)
                    -1.3223720640647498,   # j3_2 = -j2_1         (revolute)
                    1.2594757646221153      # j4_2 = j2_1 + j3_1          (revolute)
                ]
                    ),
        },

    }
}
