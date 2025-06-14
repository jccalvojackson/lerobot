from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.smolvla.configuration_smolvla import (
    PreTrainedConfig,
    SmolVLAConfig,
)
from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
)
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.scripts.control_robot import control_robot

training_dataset = LeRobotDataset(
    repo_id="jccj/so100_block_in_cup",
)

config = ControlPipelineConfig(
    robot=So100RobotConfig(),
    control=RecordControlConfig(
        repo_id="jccj/so100_block_in_cup",
        single_task="Grasp a block and put it in the cup.",
        tags=["so100", "smolvla"],
        warmup_time_s=10,
        episode_time_s=20,
        reset_time_s=10,
        fps=30,
        num_episodes=1,
        push_to_hub=False,
        policy=PreTrainedConfig.from_pretrained("artifacts/pretrained_model"),
        resume=True,
    ),
)

control_robot(config)
