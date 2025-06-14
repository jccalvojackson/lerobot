# custom record
# from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.jccj.configs import robot_config, teleoperator_config
from lerobot.record import DatasetRecordConfig, RecordConfig, _record

repo_id = "jccj/shape_matching"
# metadata_train_dataset = LeRobotDatasetMetadata("jccj/so100_block_in_cup_at_home_resized")

# pretrained_path = "jccj/smolvla_pickup_cube_resized_last"
pretrained_path = "jccj/smolvla_shape_match10k_no_aug"
pretrained_config: SmolVLAConfig = PreTrainedConfig.from_pretrained(pretrained_path)
pretrained_config.n_action_steps = 50
pretrained_config.pretrained_path = pretrained_path

eval_repo_id = "jccj/eval_shape_matching"
config = RecordConfig(
    robot=robot_config,
    dataset=DatasetRecordConfig(
        repo_id=eval_repo_id,
        single_task="pick up the square block and drop it in corresponding square shaped hole",
        fps=30,
        num_episodes=1,
        push_to_hub=False,
        episode_time_s=5 * 60,
        reset_time_s=5,
    ),
    policy=pretrained_config,
    # teleop=teleoperator_config,
    display_data=False,
    play_sounds=True,
    # resume=True,
)

_record(config)
