import json
import traceback
import os
import wandb
import importlib
import torch

from datetime import datetime

from .base.base_module import BaseModule
from .base.base_logger import BaseLogger
from .base.base_dataset import BaseDataset
from .base.base_trainer import BaseTrainer


class Orchestrator:
    def __init__(self, jobs):
        self.jobs = jobs

    def _get_wandb_config(self, config):
        # Extract parameters from job config
        wandb_config = {
            "module": config["module"]["params"],
            "dataset": config["dataset"]["params"],
            "logger": config["logger"]["params"] if "logger" in config else {},
            "trainer": config["trainer"]["params"] if "trainer" in config else {},
        }

        # Flatten the dictionary
        wandb_config = {f"{section}_{key}": value for section, params in wandb_config.items() for key, value in params.items()}

        # Merge with existing wandb config, if it exists
        if "config" in config["wandb"]:
            wandb_config = {**wandb_config, **config["wandb"]["config"]}

        return wandb_config

    def _load_class(self, class_name):
        attributes = class_name.split('.')
        current_attribute = None
        for i, attribute in enumerate(attributes):
            if i == 0:
                current_attribute = importlib.import_module(attribute)
            else:
                current_attribute = getattr(current_attribute, attribute)
        return current_attribute

    def _check_config(self, config):
        assert "name" in config, "'name' key missing in config"
        assert "wandb" in config, "'wandb' key missing in config"
        assert "project" in config["wandb"], "'project_name' missing in 'wandb' config"
        assert "api_key" in config["wandb"], "'api_key' missing in 'wandb' config"
        assert "module" in config, "'module' key missing in config"
        assert "type" in config["module"], "'type' key missing in 'module' config"
        assert "params" in config["module"], "'params' key missing in 'module' config"
        assert "dataset" in config, "'dataset' key missing in config"
        assert "type" in config["dataset"], "'type' key missing in 'dataset' config"
        assert "params" in config["dataset"], "'params' key missing in 'dataset' config"

        if "logger" in config:
            assert "params" in config["logger"], "'params' key missing in 'logger' config"
        if "trainer" in config:
            assert "params" in config["trainer"], "'params' key missing in 'trainer' config"

    def run(self):
        for job in self.jobs:

            with open(job, "r") as f:
                config = json.load(f)

            self._check_config(config)

            # Log into wandb
            api_key = config["wandb"].pop("api_key", None)
            wandb.login(key=api_key)

            name = config['name']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            job_dir = f"jobs/{name}/{timestamp}"
            os.makedirs(job_dir, exist_ok=True)

            # Save the config file
            with open(f"{job_dir}/config.json", 'w') as f:
                json.dump(config, f, indent=4)

            Logger = self._load_class(config.get("logger", {}).get("type", "deep_orchestrator.loggers.default_logger.DefaultLogger"))
            assert issubclass(Logger, BaseLogger)
            config["wandb"]["config"] = self._get_wandb_config(config)
            config["wandb"]["save_dir"] = job_dir
            logger = Logger({**config.get("logger", {}).get("params", {}), **{"job_name": name}}, config["wandb"])

            try:

                Module = self._load_class(config["module"]["type"])
                assert issubclass(Module, BaseModule)
                module = Module(config["module"]["params"])

                Dataset = self._load_class(config["dataset"]["type"])
                assert issubclass(Dataset, BaseDataset)
                dataset = Dataset(config["dataset"]["params"])

                Trainer = self._load_class(config.get("trainer", {}).get("type", "deep_orchestrator.trainers.default_trainer.DefaultTrainer"))
                assert issubclass(Trainer, BaseTrainer)
                trainer = Trainer(config.get("trainer", {}).get("params", {}))

            except Exception as e:
                logger.log_message(f"Error while preparing objects for job '{name}': {str(e)}\n{traceback.format_exc()}")

            try:
                # Prepare the data
                trainer.prepare_data(dataset)
            except Exception as e:
                logger.log_message(f"Error while preparing data for job '{name}': {str(e)}\n{traceback.format_exc()}")

            # Train the model
            try:
                trainer.train(module, logger)
                torch.save(module.state_dict(), f"{job_dir}/model.pt")
            except Exception as e:
                logger.log_message(f"Error while running job '{name}': {str(e)}\n{traceback.format_exc()}")
