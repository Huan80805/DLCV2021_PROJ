import torch
def save_checkpoint(checkpoint_path, model):
    torch.save(model.state_dict(), checkpoint_path)
    print('--------model saved to %s-------' % checkpoint_path)