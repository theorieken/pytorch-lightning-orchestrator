import json
import traceback
import os
import wandb
import importlib
import torch
import glob

from datetime import datetime

from .base.module import BaseModule
from .base.logger import BaseLogger
from .base.dataset import BaseDataset
from .base.callback import BaseCallback
from .base.trainer import BaseTrainer


def find_latest_checkpoint(directory):
    # Use glob to find all .ckpt files in the directory and its subdirectories
    checkpoint_files = glob.glob(os.path.join(directory, '**', '*.ckpt'), recursive=True)

    # If no .ckpt files are found, return None
    if not checkpoint_files:
        return None

    # Return the file with the most recent modification time
    return max(checkpoint_files, key=os.path.getmtime)


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
        assert "resume" in config, "'resume' key missing in config"
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
        if "callback" in config:
            assert "params" in config["callback"], "'params' key missing in 'callback' config"

    def run(self):
        for job in self.jobs:

            # Define default logger and trainer
            default_logger = "deep_orchestrator.loggers.csv.CsvLogger"
            default_callback = "deep_orchestrator.callbacks.default_callback.DefaultCallback"
            default_trainer = "deep_orchestrator.trainers.default_trainer.DefaultTrainer"

            # Load the config file
            with open(job, "r") as f:
                config = json.load(f)

            # Check the config file
            self._check_config(config)

            # Log into wandb and handle setup if key exists
            api_key = config["wandb"].pop("api_key", None)
            logger_conf = {}
            if api_key is not None:
                wandb.login(key=api_key)
                default_logger = "deep_orchestrator.loggers.wandb.WeightBiasesLogger"
                logger_conf = config["wandb"]
                logger_conf["config"] = self._get_wandb_config(config)

            name = config['name']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            job_dir_raw = f"jobs/{name}/"
            job_dir = f"{job_dir_raw}/{timestamp}"
            os.makedirs(job_dir, exist_ok=True)

            # Save the config file
            with open(f"{job_dir}/config.json", 'w') as f:
                json.dump(config, f, indent=4)

            Logger = self._load_class(config.get("logger", {}).get("type", default_logger))
            assert issubclass(Logger, BaseLogger)
            logger_conf["save_dir"] = job_dir
            logger = Logger(**{**config.get("logger", {}).get("params", {}), **{"job_name": name, "conf": logger_conf}})

            # Log what logger is used
            logger.log_message(f"Using logger {Logger.__name__}")

            try:

                # If resume flag is set, load the checkpoint
                latest_checkpoint = None
                if config.get('resume', False):
                    latest_checkpoint = find_latest_checkpoint(job_dir_raw)
                    if latest_checkpoint is not None:
                        logger.log_message(f"Checkpoint found. Loading trainer from checkpoint ...")
                    else:
                        logger.log_message(f"No checkpoint found in directory {job_dir}, training from scratch.")

                # Prepare all objects needed for the training
                Module = self._load_class(config["module"]["type"])
                assert issubclass(Module, BaseModule), "Module must be a subclass of BaseModule"
                module = Module(**config["module"]["params"])

                Dataset = self._load_class(config["dataset"]["type"])
                assert issubclass(Dataset, BaseDataset), "Dataset must be a subclass of BaseDataset"
                dataset = Dataset(**config["dataset"]["params"])

                Trainer = self._load_class(config.get("trainer", {}).get("type", default_trainer))
                assert issubclass(Trainer, BaseTrainer), "Trainer must be a subclass of BaseTrainer"
                trainer = Trainer(**config.get("trainer", {}).get("params", {}))

                Callback = self._load_class(config.get("callback", {}).get("type", default_callback))
                assert issubclass(Callback, BaseCallback), "Callback must be a subclass of BaseCallback"
                callback = Callback(config.get("callback", {}).get("params", {}), logger)

                try:
                    trainer.prepare_data(dataset)
                    trainer.prepare_trainer(logger, callback)
                    try:
                        trainer.train(module, latest_checkpoint)
                        torch.save(module.state_dict(), f"{job_dir}/model.pt")
                    except Exception as e:
                        logger.log_message(f"Error while running job '{name}': {str(e)}\n{traceback.format_exc()}", kind="error")

                except Exception as e:
                    logger.log_message(f"Error while preparing trainer for job '{name}': {str(e)}\n{traceback.format_exc()}", kind="error")

            except Exception as e:
                logger.log_message(f"Error while preparing objects for job '{name}': {str(e)}\n{traceback.format_exc()}", kind="error")
