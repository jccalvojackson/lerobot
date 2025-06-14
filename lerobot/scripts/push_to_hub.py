# push to hub
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig

repo_id = "jccj/shape_matching"
metadata_train_dataset = LeRobotDatasetMetadata(repo_id)

episodes_to_skip = [40, 54]

episodes_to_keep = [i for i in metadata_train_dataset.episodes if i not in episodes_to_skip]

dataset = LeRobotDataset(
    repo_id=repo_id,
    tolerance_s=1e-4,
)

print(f"Episodes to keep: {episodes_to_keep}")

filtered_dataset_repo_id = f"{repo_id}_filtered"
new_dataset = LeRobotDataset.create(
    filtered_dataset_repo_id,
    metadata_train_dataset.fps,
    root=metadata_train_dataset.root,
    robot_type=metadata_train_dataset.robot_type,
    features=metadata_train_dataset.features,
    use_videos=True,
    image_writer_processes=metadata_train_dataset.image_writer_processes,
)
# dataset = LeRobotDataset(
#     repo_id=repo_id,
#     tolerance_s=1e-4,
#     episodes=episodes_to_keep,
# )


# dataset.push_to_hub(
#     tags=["lerobot", "so100", "match_shape"],
#     private=False,
# )
