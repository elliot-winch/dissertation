import torch.optim.lr_scheduler as lrs

def get_scheduler(config, optimizer):

    if hasattr(config, 'learning_rate_scheduler'):
        if config.learning_rate_scheduler == "ReduceLROnPlateau":
            return lrs.ReduceLROnPlateau(optimizer, patience=config.learning_rate_patience)

    return None
