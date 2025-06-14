from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig

top_camera_config = OpenCVCameraConfig(
    index_or_path=1,
    color_mode=ColorMode.RGB,
    # Note: Using camera's reliable default resolution to avoid inconsistent behavior
    width=640,
    height=480,
    fps=30,
)

wrist_left_camera_config = OpenCVCameraConfig(
    index_or_path=0,
    color_mode=ColorMode.RGB,
    width=640,
    height=480,
    fps=30,
)

camera_config = {
    "top": top_camera_config,
    "wrist": wrist_left_camera_config,
}
