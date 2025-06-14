from lerobot.jccj.configs import robot_config, teleoperator_config
from lerobot.teleoperate import TeleoperateConfig, _teleoperate

teleoperate_config = TeleoperateConfig(
    teleop=teleoperator_config,
    robot=robot_config,
    fps=30,
    teleop_time_s=15 * 60,
    display_data=False,
)

_teleoperate(teleoperate_config)
