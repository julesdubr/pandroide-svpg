from salina import instantiate_class

"""
- Setup the logger for the visualization of the results
- Using the logging mechanism provided under the hood by salina (tensorboard) and the arguments for the
configuration file
"""


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, epoch):
        self.logger.add_scalar(log_string, loss.item(), epoch)

    # Log losses
    def log_losses(self,epoch, critic_loss, entropy_loss, policy_loss):
        self.add_log("critic_loss", critic_loss, epoch)
        self.add_log("entropy_loss", entropy_loss, epoch)
        self.add_log("policy_loss", policy_loss, epoch)