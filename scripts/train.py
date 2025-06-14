from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.train import _train

repo_id = "lerobot/pusht"


# policy_config = SmolVLAConfig()
policy_config = DiffusionConfig()

dataset_config = DatasetConfig(
    repo_id=repo_id,
    image_transforms=ImageTransformsConfig(
        enable=False,
    ),
)

train_config = TrainPipelineConfig(
    dataset=dataset_config,
    policy=policy_config,
    batch_size=64,
    steps=200_000,
    save_freq=5000,
    wandb=WandBConfig(
        enable=True,
    ),
)

if __name__ == "__main__":
    init_logging()
    _train(train_config)
