from lerobot.common.envs.configs import EnvTransformConfig, HILSerlRobotEnvConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.jccj.configs.robot import robot_config, robot_end_effector_config
from lerobot.jccj.configs.teleoperator import teleoperator_config
from lerobot.scripts.rl.gym_manipulator import _main

config = HILSerlRobotEnvConfig(
    robot=robot_end_effector_config,
    teleop=teleoperator_config,
    wrapper=EnvTransformConfig(
        display_cameras=False,
        add_joint_velocity_to_observation=True,
        add_current_to_observation=True,
        add_ee_pose_to_observation=True,
        crop_params_dict={
            "observation.images.top": [270, 170, 90, 190],
            "observation.images.wrist": [0, 0, 480, 480],
        },
        resize_size=(128, 128),
        control_time_s=20.0,
        use_gripper=True,
        gripper_penalty=-0.02,
        gripper_penalty_in_reward=False,
        fixed_reset_joint_positions=[
            0.0,
            0.0,
            0.0,
            90.0,
            0.0,
            5.0,
        ],
        reset_time_s=2.5,
        control_mode="leader",
    ),
    name="real_robot",
    mode="record",
    repo_id="jccj/hilserlrobotenv",
    task="shape_sorter",
    num_episodes=15,
    episode=0,
    pretrained_policy_name_or_path=None,
    device="mps",
    push_to_hub=True,
    fps=10,
    features={
        "observation.images.top": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 128, 128),
        ),
        "observation.images.wrist": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 128, 128),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(15,),
        ),
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(3,),
        ),
    },
    features_map={
        "observation.images.top": "observation.images.side",
        "observation.images.wrist": "observation.images.wrist",
        "observation.state": "observation.state",
        "action": "action",
    },
    reward_classifier_pretrained_path=None,
)

_main(config)
