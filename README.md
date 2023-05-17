# Pytorch Lightning Orchestrator

This project creates a wrapper around PyTorch lighting which orchestrates different lightning modules and connects them to wandb.

## Installation

Simply install the orchestrator via pip:

```bash
pip install lightning-orchestrator
```

## Usage

The orchestrator is used by creating a list of jobs which are then run in sequence. Each job is a dictionary with the following keys:

```python
from deep_orchestrator import Orchestrator

jobs = ['job1', 'job2', 'job3']

orchestrator = Orchestrator(jobs=jobs)
orchestrator.run()
```

Jobs are encoded like this

```json
{
    "name": "job1",
    "presume": true,
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
            "param1": "value1",
            "param2": "value2"
        }
    },
    "logger": {
        "type": "YourLogger",
        "params": {
            "param1": "value1",
            "param2": "value2"
        }
    }
}

```