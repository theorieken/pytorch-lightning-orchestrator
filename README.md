# Pytorch Lightning Orchestrator

This project creates a wrapper around PyTorch lightning which orchestrates different lightning modules and connects them to Weights & Biases (wandb). It allows you to specify custom datasets, modules, loggers, and trainers, with support for default implementations if none are provided.

## Installation

Simply install the orchestrator via pip:

```bash
pip install lightning-orchestrator
```

## Usage

The orchestrator is used by creating a list of jobs which are then run in sequence. Each job is a dictionary with the following keys:

```python
from deep_orchestrator import Orchestrator

jobs = ['job1.json', 'job2.json', 'job3.json']

orchestrator = Orchestrator(jobs=jobs)
orchestrator.run()
```

Jobs are encoded as JSON files with a structure like the following:

```json 
{
    "name": "job1",
    "resume": true,
    "wandb": {
        "api_key": "your-wandb-api-key",
        "project_name": "your-wandb-project-name"
    },
    "module": {
        "type": "YourLightningModule",
        "params": {
            "param1": "value1",
            "param2": "value2"
        }
    },
    "dataset": {
        "type": "YourDataset",
        "params": {
            "path": "/path/to/dataset",
            "split": 0.8,
            "fraction": 1.0
        }
    },
    "logger": {
        "type": "YourLogger",
        "params": {
            "param1": "value1",
            "param2": "value2"
        }
    },
    "trainer": {
        "type": "YourTrainer",
        "params": {
            "max_epochs": 1000,
            "batch_size": 4,
            "shuffle": true,
            "num_workers": 4,
            "pin_memory": true
        }
    }
}
```

The orchestrator will load your custom classes dynamically based on the "type" provided. If no custom class is specified, the orchestrator will use a default implementation. For example, if you don't specify a "trainer", the orchestrator will use the DefaultTrainer.

The params field in each section allows you to provide parameters for your custom classes. For example, in the "dataset" section, you can specify the path to your dataset, the fraction of the dataset to use, and the split ratio for the training and validation sets.

The "wandb" section is used to specify parameters for Weights & Biases logging, such as your API key and the project name. If you set "presume" to true, the orchestrator will continue to the next job if an error occurs.

The orchestrator will log all outputs to a file named "log.txt" in a directory named after the job. It will also save the final model weights to this directory.
