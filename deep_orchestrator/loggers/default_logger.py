from deep_orchestrator.base.logger import BaseLogger


class DefaultLogger(BaseLogger):
    def on_step(self, trainer, pl_module, outputs, batch):
        pass

    def on_epoch(self, trainer, pl_module, batch):
        pass
