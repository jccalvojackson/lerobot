#!/usr/bin/env python
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.envs.configs import (
    EEActionSpaceConfig,
    EnvWrapperConfig,
    HILEnvConfig,
    VideoRecordConfig,
)
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.server.gym_manipulator import make_robot_env, record_dataset


class PickCubeExpertPolicy:
    """Expert policy for pick cube task with different stages."""

    def __init__(self, robot):
        self.stage_sequence = ["hover", "stabilize", "grasp", "lift"]
        self.current_stage_idx = 0
        self.steps_in_stage = 0
        self.steps_per_stage = 40  # 40 steps per stage

    def reset(self):
        """Reset policy state."""
        self.current_stage_idx = 0
        self.steps_in_stage = 0

    @property
    def current_stage(self):
        return self.stage_sequence[self.current_stage_idx]

    def select_action(self, observation):
        """
        Generate expert action based on current stage and observation.
        Returns actions of shape (4,) on the specified device.
        """
        # need to determine the current stage from the observation
        # Extract state from observation
        state = observation["observation.state"].squeeze(0)  # Remove batch dimension if present

        # Update stage if needed
        self.steps_in_stage += 1
        if self.steps_in_stage >= self.steps_per_stage:
            self.steps_in_stage = 0
            self.current_stage_idx = min(self.current_stage_idx + 1, len(self.stage_sequence) - 1)

        # Estimate cube position from camera observation
        # In a real implementation, this would use visual detection
        # For this example, we use a fixed target position
        cube_pos = torch.tensor([0.35, 0.0, 0.05], device=self.device)

        # Stage-dependent targets (delta-based control)
        if self.current_stage == "hover":
            # Move above the cube
            target_pos = cube_pos + torch.tensor([0.0, 0.0, 0.1], device=self.device)
            gripper = torch.tensor([2.0], device=self.device)  # Open gripper (2.0)
        elif self.current_stage == "stabilize":
            # Hold position above the cube
            target_pos = cube_pos + torch.tensor([0.0, 0.0, 0.1], device=self.device)
            gripper = torch.tensor([2.0], device=self.device)  # Keep gripper open
        elif self.current_stage == "grasp":
            # Move down to grasp
            target_pos = cube_pos + torch.tensor([0.0, 0.0, 0.03], device=self.device)
            gripper = torch.tensor([0.0], device=self.device)  # Close gripper (0.0)
        elif self.current_stage == "lift":
            # Lift cube
            target_pos = cube_pos + torch.tensor([0.0, 0.0, 0.2], device=self.device)
            gripper = torch.tensor([0.0], device=self.device)  # Keep gripper closed

        # Compute position delta (assuming end-effector position is part of state)
        # In a real robot implementation, this would use proper state estimation
        ee_pos = state[-3:] if len(state) >= 3 else torch.zeros(3, device=self.device)
        pos_delta = target_pos - ee_pos

        # Scale deltas to fit action space
        scaled_delta = torch.clamp(pos_delta, -0.025, 0.025)
        action = torch.cat([scaled_delta, gripper], dim=0)

        return action


def record_dataset_with_expert(config):
    """Record a dataset using the expert policy for pick cube task."""
    # Create environment using the make_robot_env function from gym_manipulator.py
    env = make_robot_env(config)

    # Initialize expert policy
    policy = PickCubeExpertPolicy(device=config.device)

    # Setup features for dataset based on environment observation space
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": config.features["observation.state"].shape,
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),  # x, y, z, gripper
            "names": ["delta_x", "delta_y", "delta_z", "gripper"],
        },
        "next.reward": {"dtype": "float32", "shape": (1,)},
        "next.done": {"dtype": "bool", "shape": (1,)},
    }

    # Add image features
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": tuple(env.observation_space[key].shape[1:]),  # Remove batch dimension
            }

    # Create dataset
    dataset = LeRobotDataset.create(
        config.repo_id,
        config.fps,
        root=config.dataset_root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    # Record episodes
    episode_count = 0
    while episode_count < config.num_episodes:
        print(f"\n🎬 Starting episode {episode_count + 1}/{config.num_episodes}")

        # Reset environment and policy
        obs, info = env.reset()
        policy.reset()

        # Track success state collection
        success_detected = False
        success_steps_collected = 0

        frames_collected = 0
        keep_episode = False

        # Run episode steps until terminated or max steps reached
        while frames_collected < config.episode_length:
            # Get action from policy
            action = policy.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Check for success
            if reward > 0:
                success_detected = True
                keep_episode = True
                print(f"✅ Success detected at step {frames_collected}!")

            # Process observation for dataset
            obs_processed = {k: v.cpu().squeeze(0).numpy() for k, v in obs.items()}

            # Prepare frame data
            frame = {
                **obs_processed,
                "action": action.cpu().numpy(),
                "next.reward": np.array([reward], dtype=np.float32),
                "next.done": np.array([terminated or truncated], dtype=bool),
                "task": "pick cube",
            }

            # Add frame to dataset
            dataset.add_frame(frame)
            frames_collected += 1

            # Update observation for next step
            obs = next_obs

            # If successful, check if we've collected enough additional steps
            if success_detected:
                success_steps_collected += 1
                if success_steps_collected >= config.number_of_steps_after_success:
                    print(f"Collected {success_steps_collected} additional success steps. Ending episode.")
                    break

            # Check if episode has terminated
            if terminated or truncated:
                break

            # Sleep to maintain FPS (if needed)
            time.sleep(1.0 / config.fps)

        # Save or discard episode
        if keep_episode:
            dataset.save_episode()
            episode_count += 1
            print(f"✅ Episode {episode_count} saved with {frames_collected} frames")
        else:
            dataset.clear_episode_buffer()
            print(f"🚫 Episode discarded - no success detected")

    print(f"Dataset recording complete. Recorded {episode_count} episodes.")
    if config.push_to_hub:
        print("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub()


if __name__ == "__main__":
    from configs.gym_hil_env import config

    # Record dataset
    record_dataset_with_expert(config)
