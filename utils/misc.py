import numpy as np
import torch

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]  # we support only one param_group
    assert(len(lr)==1) 
    lr = lr[0]
    return lr


def load_checkpoint(path, model, device="cuda:0", optimizer=None, scheduler=None):
    """
    https://pytorch.org/docs/stable/generated/torch.load.html
    When you call torch.load() on a file which contains GPU tensors, those tensors will be loaded to GPU by default. 
    You can call torch.load(.., map_location='cpu') and then load_state_dict() to avoid GPU RAM 
    surge when loading a model checkpoint.
    """

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if optimizer:
        if ckpt["optimizer_state_dict"]:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler:
        if ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt["epoch"] if "epoch" in ckpt else 0
    step = ckpt["step"] if "step" in ckpt else 0
    loss = ckpt["loss"] if "loss" in ckpt else float("inf")

    return epoch, step, loss


def save_checkpoint(
    path, model, 
    optimizer=None, scheduler=None, 
    epoch=None, step=None, loss=None):

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "loss": loss
    }
    torch.save(ckpt, path)


def concat_img(img_list):
    h = max([img.shape[0] for img in img_list]) 
    w = sum([img.shape[1] for img in img_list])
    result_img = np.zeros((h, w, 3))
    start_h, start_w = 0, 0
    for img in img_list:
        result_img[start_h:start_h+img.shape[0], start_w:start_w+img.shape[1], :] = img
        start_w += img.shape[1] 
    return result_img