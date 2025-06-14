from enum import Enum

from lerobot.calibrate import CalibrateConfig, _calibrate
from lerobot.jccj.configs import robot_config, teleoperator_config


class Device(Enum):
    ROBOT = "robot"
    TELEOPERATOR = "teleoperator"


def get_calibrate_config(device: Device) -> CalibrateConfig:
    if device == Device.ROBOT:
        return CalibrateConfig(robot=robot_config)
    elif device == Device.TELEOPERATOR:
        return CalibrateConfig(teleop=teleoperator_config)
    else:
        raise ValueError(f"Invalid device: {device}")


def main(device: Device):
    _calibrate(get_calibrate_config(device))


if __name__ == "__main__":
    import typer

    typer.run(main)
