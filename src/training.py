import os
import hydra
import logging
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="train", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Main training function that supports multiple trainer types

    Parameters:
         cfg (DictConfig): Hydra configuration object
    """
    # Set environment variables from config
    if 'env' in cfg:
        for key, value in cfg.env.items():
            os.environ[key] = str(value)
    
    # Setup logging
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.info("Starting training...")
    
    # Instantiate trainer from config
    trainer = instantiate(cfg.trainer)
    trainer.set_logger(logger)
    
    # Start training
    trainer.train()
        

if __name__ == "__main__":
    run()
