from pathlib import Path

from lerobot.common.robots.so100_follower import (
    SO100Follower,
    SO100FollowerConfig,
    SO100FollowerEndEffectorConfig,
)

from .cameras import camera_config

robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58FA1015951",
    id="follower",
    cameras=camera_config,
)

# Max ee position [0.4006, 0.3076, 0.3319]
# Min ee position [0.1901, -0.1089, 0.1279]
# Max joint pos position [22.4775, 38.3861, 56.1737, 100.0, 24.1026, 81.759]
# Min joint pos position [-50.8492, -29.8261, -48.8531, -28.322, -22.0513, 2.443]
robot_end_effector_config = SO100FollowerEndEffectorConfig(
    port="/dev/tty.usbmodem58FA1015951",
    id="follower_end_effector",
    cameras=camera_config,
    end_effector_bounds={
        "min": [-0.0391, -0.2774, 0.0928],
        "max": [0.3932, 0.3837, 0.328],
    },
    end_effector_step_sizes={
        "x": 0.025,
        "y": 0.025,
        "z": 0.025,
    },
)

if __name__ == "__main__":
    # robot = SO100Follower(robot_config)
    robot.connect()
    robot.disconnect()
