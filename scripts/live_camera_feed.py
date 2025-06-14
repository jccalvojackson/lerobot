#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Live camera feed viewer for all detected cameras.

Shows real-time video feeds from all available cameras in separate windows.
Press 'q' in any window or Ctrl+C to exit.

Example:

```shell
python scripts/live_camera_feed.py
```

```shell
python scripts/live_camera_feed.py realsense
```
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Add the parent directory to the path to import lerobot modules
sys.path.append(str(Path(__file__).parent.parent))

from lerobot.common.cameras.configs import ColorMode
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.common.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,
)

logger = logging.getLogger(__name__)


def find_all_opencv_cameras() -> List[Dict[str, Any]]:
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: List[Dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> List[Dict[str, Any]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: List[Dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> List[Dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense" or "opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: List[Dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def create_camera_instance(cam_meta: Dict[str, Any]) -> Dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=int(cam_id),
                color_mode=ColorMode.RGB,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=False)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def cleanup_cameras(cameras_to_use: List[Dict[str, Any]]):
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def display_live_feeds(camera_type: str | None = None):
    """
    Connects to detected cameras (optionally filtered by type) and displays live feeds.
    Uses default stream profiles for width, height, and FPS.

    Args:
        camera_type: Optional string to filter cameras ("realsense" or "opencv").
                     If None, uses all detected cameras.
    """
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot display feeds.")
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta)
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting live feed display.")
        return

    logger.info(f"Starting live feed display from {len(cameras_to_use)} cameras.")
    logger.info("Press 'q' in any window or Ctrl+C to exit.")

    # Create window names for each camera
    window_names = []
    for i, cam_dict in enumerate(cameras_to_use):
        meta = cam_dict["meta"]
        cam_type_str = str(meta.get("type", "unknown"))
        cam_id_str = str(meta.get("id", "unknown"))
        window_name = f"{cam_type_str} Camera {cam_id_str}"
        window_names.append(window_name)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Read from all cameras and display
            for i, (cam_dict, window_name) in enumerate(zip(cameras_to_use, window_names)):
                try:
                    # Read image data
                    image_data = cam_dict["instance"].read()

                    # Convert RGB to BGR for OpenCV display
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        # Convert RGB to BGR
                        image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = image_data

                    # Display the image
                    cv2.imshow(window_name, image_bgr)

                except TimeoutError:
                    logger.warning(f"Timeout reading from camera {i}")
                    continue
                except Exception as e:
                    logger.error(f"Error reading from camera {i}: {e}")
                    continue

            # Check for 'q' key press or window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("'q' key pressed. Exiting...")
                break

            # Check if any window was closed
            try:
                # This is a simple way to check if windows still exist
                # If getWindowProperty returns -1, the window was closed
                for window_name in window_names:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        logger.info(f"Window '{window_name}' was closed. Exiting...")
                        raise KeyboardInterrupt
            except cv2.error:
                # Window might have been closed
                logger.info("A window was closed. Exiting...")
                break

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Live feed interrupted by user.")
    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        cleanup_cameras(cameras_to_use)
        logger.info("Live feed display finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live camera feed viewer for all detected cameras.")

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Specify camera type to display ('realsense', 'opencv'). Shows all cameras if omitted.",
    )

    args = parser.parse_args()
    display_live_feeds(args.camera_type)
