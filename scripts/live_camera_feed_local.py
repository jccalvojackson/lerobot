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
Live camera feed viewer using predefined camera configuration with local OpenCVCamera implementation.

Shows real-time video feeds from configured cameras in separate windows.
Press 'q' in any window or Ctrl+C to exit.

Example:

```shell
python scripts/live_camera_feed_local.py
```
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2

# Add the parent directory to the path to import lerobot modules
sys.path.append(str(Path(__file__).parent.parent))

from lerobot.jccj.configs.cameras import camera_config

# Import OpenCVCamera from local tmp.py file
from scripts.tmp import OpenCVCamera

logger = logging.getLogger(__name__)


def create_camera_instances() -> Dict[str, Dict[str, Any]]:
    """Create camera instances from the predefined configuration."""
    cameras = {}

    print("\n--- Configured Cameras ---")
    for camera_name, config in camera_config.items():
        print(f"Camera: {camera_name}")
        print(f"  Index/Path: {config.index_or_path}")
        print(f"  Resolution: {config.width}x{config.height}")
        print(f"  FPS: {config.fps}")
        print(f"  Color Mode: {config.color_mode}")
        print("-" * 20)

        try:
            logger.info(f"Connecting to {camera_name} camera...")
            instance = OpenCVCamera(config)
            instance.connect()
            cameras[camera_name] = {"instance": instance, "config": config}
            logger.info(f"Successfully connected to {camera_name} camera")
        except Exception as e:
            logger.error(f"Failed to connect to {camera_name} camera: {e}")
            continue

    return cameras


def cleanup_cameras(cameras: Dict[str, Dict[str, Any]]):
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras)} cameras...")
    for camera_name, cam_dict in cameras.items():
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
                logger.info(f"Disconnected {camera_name} camera")
        except Exception as e:
            logger.error(f"Error disconnecting {camera_name} camera: {e}")


def display_live_feeds():
    """
    Connects to configured cameras and displays live feeds.
    """
    cameras = create_camera_instances()

    if not cameras:
        logger.warning("No cameras could be connected. Aborting live feed display.")
        return

    logger.info(f"Starting live feed display from {len(cameras)} cameras.")
    logger.info("Press 'q' in any window or Ctrl+C to exit.")

    # Create window names for each camera
    window_names = {}
    for camera_name in cameras.keys():
        window_name = f"{camera_name.replace('_', ' ').title()} Camera"
        window_names[camera_name] = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Read from all cameras and display
            for camera_name, cam_dict in cameras.items():
                window_name = window_names[camera_name]
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
                    logger.warning(f"Timeout reading from {camera_name} camera")
                    continue
                except Exception as e:
                    logger.error(f"Error reading from {camera_name} camera: {e}")
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
                for window_name in window_names.values():
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
        cleanup_cameras(cameras)
        logger.info("Live feed display finished.")


if __name__ == "__main__":
    display_live_feeds()
