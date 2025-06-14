from lerobot.jccj.configs.robot import robot_config
from lerobot.jccj.configs.teleoperator import teleoperator_config
from lerobot.scripts.find_joint_limits import (
    FindJointLimitsConfig,
    _find_joint_and_ee_bounds,
)

config = FindJointLimitsConfig(
    teleop=teleoperator_config,
    robot=robot_config,
)

_find_joint_and_ee_bounds(config)
