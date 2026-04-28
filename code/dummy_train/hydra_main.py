"""Build the dataclass from Hydra-managed configuration."""

from __future__ import annotations

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from common import ExperimentConfig, train


cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Convert the Hydra config into the shared dataclass and hand it to train()."""
    config = ExperimentConfig(**OmegaConf.to_container(cfg.experiment, resolve=True))
    train(config)


if __name__ == "__main__":
    main()
