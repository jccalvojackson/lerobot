from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# repo_id = "jccj/so100_block_in_cup_at_home_cropped_resized"
repo_id = "jccj/so100_block_in_cup_at_home"

dataset = LeRobotDataset(repo_id, episodes=[0])

top_camera_key = dataset.meta.camera_keys[0]
wrist_camera_key = dataset.meta.camera_keys[1]

episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

frames_top = [dataset[idx][top_camera_key] for idx in range(from_idx, to_idx)]
frames_wrist = [dataset[idx][wrist_camera_key] for idx in range(from_idx, to_idx)]
