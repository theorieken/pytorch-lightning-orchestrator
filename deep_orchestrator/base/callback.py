from pytorch_lightning.callbacks import Callback


class BaseCallback(Callback):

    def __init__(self, params, logger=None):
        assert 'logger' is not None, 'Logger needs to be passed to the callback'
        self.logger = logger
        self.params = params

    def setup(self, trainer, pl_module, stage):
        self.logger.log_message('Preparing the model for {}...'.format(stage), kind='info')

    def teardown(self, trainer, pl_module, stage):
        self.logger.log_message('Finishing up the model from {}'.format(stage), kind='info')
