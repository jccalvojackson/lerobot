from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.common.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

camera_config = {
    "iphone": OpenCVCameraConfig(
        index_or_path=0,
        width=1920,
        height=1080,
    ),
    # "laptop": OpenCVCameraConfig(
    #     index_or_path=1,
    #     width=1920,
    #     height=1080,
    # ),
}

robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58FA1015951",
    id="follower",
    cameras=camera_config,
)

teleop_config = SO100LeaderConfig(
    port="/dev/tty.usbmodem58FD0171191",
    id="leader",
)
