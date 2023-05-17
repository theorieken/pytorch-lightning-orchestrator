import json
import traceback
import os
import wandb
import pytorch_lightning as pl
import importlib
import torch
from rich.progress import Progress

from .base.base_module import BaseModule
from .base.base_logger import BaseLogger
from .base.base_dataset import BaseDataset


class Orchestrator:
    def __init__(self, jobs):
        self.jobs = jobs
        self.progress = Progress()

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
        assert "logger" in config, "'logger' key missing in config"
        assert "type" in config["logger"], "'type' key missing in 'logger' config"
        assert "params" in config["logger"], "'params' key missing in 'logger' config"

    def run(self):
        with self.progress:
            task = self.progress.add_task("[cyan]Training...", total=len(self.jobs))
            for job in self.jobs:
                with open(job, "r") as f:
                    config = json.load(f)

                self._check_config(config)

                # Log into wandb
                api_key = config["wandb"].pop("api_key", None)
                wandb.login(key=api_key)

                name = config['name']
                os.makedirs(f"Jobs/{name}", exist_ok=True)

                Module = self._load_class(config["module"]["type"])
                assert issubclass(Module, BaseModule)
                module = Module(config["module"]["params"])

                Logger = self._load_class(config["logger"]["type"])
                assert issubclass(Logger, BaseLogger)
                logger = Logger({**config["logger"]["params"], **{"job_name": name}}, config["wandb"])

                Dataset = self._load_class(config["dataset"]["type"])
                assert issubclass(Dataset, BaseDataset)
                dataset = Dataset(config["dataset"]["params"])

                try:
                    trainer = pl.Trainer(logger=logger)
                    trainer.fit(module, dataset)
                    torch.save(module.state_dict(), f"Jobs/{name}/model.pt")
                except Exception as e:
                    logger.log_message(f"Error while running job '{name}': {str(e)}\n{traceback.format_exc()}")
                self.progress.advance(task)
