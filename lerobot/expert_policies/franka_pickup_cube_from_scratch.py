from dataclasses import dataclass
from enum import Enum
from typing import Any

import gym_hil.mujoco_gym_env
import gymnasium as gym
import mujoco
import numpy as np
import torch
from gym_hil.envs import PandaPickCubeGymEnv
from gymnasium import spaces

Robot = Any


class PickCubeStage(str, Enum):
    HOVER = "hover"
    GRASP = "grasp"
    LIFT = "lift"
    DONE = "done"


UNIT_ZETTA = np.array([0, 0, 1])


@dataclass
class PickCubeExpertPolicy:
    base_environment: PandaPickCubeGymEnv
    delta_size: float = 0.01
    """This policy just moves towards the target with a step size of delta_size"""

    def __post_init__(self):
        self.box_size = self.base_environment._block_z
        self.open_gripper_position = np.array([0])  # closed is 2
        self.close_gripper_position = np.array([2])

    def _non_hover_position_and_size(self) -> tuple[np.ndarray, np.ndarray]:
        _cube_position = self.get_cube_position()
        cube_position = self._cube_to_floor(_cube_position)
        non_hover_height = 3 * 2 * self.box_size
        body_center = cube_position - self.box_size * UNIT_ZETTA + non_hover_height / 2 * UNIT_ZETTA
        delta_size = np.array([self.box_size, self.box_size, non_hover_height / 2])
        return body_center, delta_size

    def get_non_hover_space(self) -> tuple[spaces.Box, np.ndarray]:
        non_hover_position, delta_size = self._non_hover_position_and_size()
        # add this box to the model
        return spaces.Box(
            low=non_hover_position - delta_size,
            high=non_hover_position + delta_size,
            dtype=np.float64,
        ), non_hover_position

    def get_gripper_position(self) -> np.ndarray:
        return self.base_environment._data.sensor("2f85/pinch_pos").data

    def get_cube_position(self) -> np.ndarray:
        return self.base_environment.unwrapped._data.jnt("block").qpos[:3]

    def get_hover_position(self) -> np.ndarray:
        _cube_position = self.get_cube_position()
        cube_position = self._cube_to_floor(_cube_position)
        return cube_position + 2 * 2 * self.box_size * UNIT_ZETTA

    def get_gripper_pose(self) -> np.ndarray:
        MAX_GRIPPER_POSE = 255
        return self.base_environment.get_gripper_pose() / MAX_GRIPPER_POSE * 2

    def gripper_closed(self) -> bool:
        return np.isclose(self.get_gripper_pose(), self.close_gripper_position)

    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        current_gripper_position = self.get_gripper_position()
        box_space, cube_position = self.get_cube_box_space()
        if box_space.contains(current_gripper_position) and self.gripper_closed():
            lift_position = self.get_lift_position(cube_position)
            return self.lift_action(lift_position, current_gripper_position)
        non_hover_space, _ = self.get_non_hover_space()
        if not non_hover_space.contains(current_gripper_position):
            hover_position = self.get_hover_position()
            return self.hover_action(hover_position, current_gripper_position)
        if not box_space.contains(current_gripper_position):
            return self.hover_action(cube_position, current_gripper_position)
        # then the gripper must be open
        assert not self.gripper_closed(), "Gripper is closed"
        return self.grasp_action()

        raise ValueError("Invalid state")

    def get_lift_position(self, _cube_position: np.ndarray) -> np.ndarray:
        # independent of current height
        cube_position = self._cube_to_floor(_cube_position)
        return cube_position + 4 * 2 * self.box_size * UNIT_ZETTA

    def _cube_to_floor(self, cube_position: np.ndarray) -> np.ndarray:
        _cube_position = cube_position.copy()
        _cube_position[-1] = self.box_size
        return _cube_position

    def get_cube_box_space(self) -> tuple[spaces.Box, np.ndarray]:
        cube_position = self.get_cube_position()
        return spaces.Box(
            low=cube_position - self.box_size,
            high=cube_position + self.box_size,
            dtype=np.float64,
        ), cube_position

    def hover_action(self, hover_position: np.ndarray, current_gripper_position: np.ndarray) -> torch.Tensor:
        target_position = hover_position - current_gripper_position
        return self.navigate_to_position(
            target_position,
            grip=self.open_gripper_position,
        )

    def navigate_to_position(self, target_position: np.ndarray, grip: np.ndarray) -> torch.Tensor:
        norm = np.linalg.norm(target_position)
        if norm < 1e-2:
            return torch.tensor([0, 0, 0, grip.item()], dtype=torch.float32)
        unit_vector = target_position / norm
        delta_vector = unit_vector * self.delta_size
        return torch.tensor(np.concatenate([delta_vector, grip]), dtype=torch.float32)

    def grasp_action(self) -> torch.Tensor:
        # if Im grasping the cube, return trivial action
        return torch.tensor([0, 0, 0, self.close_gripper_position.item()], dtype=torch.float32)

    def lift_action(self, lift_position: np.ndarray, gripper_position: np.ndarray) -> torch.Tensor:
        return self.navigate_to_position(
            lift_position - gripper_position,
            grip=self.close_gripper_position,
        )


def add_visualization(
    env: gym.Env,
    position: np.ndarray,
    size: np.ndarray,
    gripper_position: np.ndarray | None = None,
    box_index: int | None = None,
):
    viewer = env.env.env.env.env.env.env.env.env.env.env._viewer
    scene = viewer.user_scn
    box_index = update_shape(
        position,
        size,
        scene,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        index=box_index,
    )
    if gripper_position is not None:
        _ = update_shape(
            gripper_position,
            np.array([0.01, 0.01, 0.01]),
            scene,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            index=box_index + 1,
        )
    viewer.sync()
    return box_index


def update_shape(
    position,
    size,
    scene,
    type: mujoco.mjtGeom,
    index: int | None = None,
):
    geom_to_add = scene.geoms[scene.ngeom] if index is None else scene.geoms[index]
    rect_mat_dynamic = np.identity(3).flatten()
    rect_rgba_dynamic = np.array([1.0, 0.0, 0.0, 0.7])  # Red, 70% transparent
    if index is None:
        mujoco.mjv_initGeom(
            geom=geom_to_add,
            type=type,
            size=size,
            pos=position,
            mat=rect_mat_dynamic,
            rgba=rect_rgba_dynamic,
        )
        geom_to_add.category = mujoco.mjtCatBit.mjCAT_DECOR  # Visual decoration
        geom_to_add.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        geom_to_add.objid = -1  # Not tied to a model object
    else:
        geom_to_add.pos = position
        geom_to_add.size = size
    if index is None:
        scene.ngeom += 1
        return scene.ngeom - 1
    return index


if __name__ == "__main__":
    import numpy as np

    from configs.gym_hil_env import config
    from lerobot.scripts.server.gym_manipulator import make_robot_env, record_dataset

    env0 = make_robot_env(config)

    base_environment: PandaPickCubeGymEnv = env0.unwrapped

    env = env0
    policy = PickCubeExpertPolicy(
        base_environment,
        delta_size=0.005,
    )

    record_dataset(
        env,
        policy,
        config,
    )
