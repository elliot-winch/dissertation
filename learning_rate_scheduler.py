import torch.optim.lr_scheduler as lrs

def get_scheduler(config, optimizer):

    if hasattr(config, 'learning_rate_scheduler'):
        if config.learning_rate_scheduler == "ReduceLROnPlateau":
            return lrs.ReduceLROnPlateau(optimizer, patience=config.learning_rate_patience)
        if config.learning_rate_scheduler == "ExponentialLR":
            return lrs.ExponentialLR(optimizer, gamma=config.learning_rate_gamma)

    return None

def step_scheduler(scheduler, config, validation_loss):
    if scheduler is not None:
        if config.learning_rate_scheduler == "ReduceLROnPlateau":
            scheduler.step(validation_loss)
        else:
            scheduler.step()
