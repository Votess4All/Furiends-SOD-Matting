import glob
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter
from u2net_train_aim2k import make_train_dataloader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train():
    wandb.init(project="furiends", entity="pengyuyan")

    wandb.define_metric("train/global_step")
    # wandb.define_metric("val/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    # wandb.define_metric("val/*", step_metric="val/global_step")

    bs = 16         # batch size
    lr = 0.01       # learn rate
    epochs = 40     # total epochs
    save_feq = 10

    # modnet = torch.nn.DataParallel(MODNet()).cuda()
    device = torch.device('cuda:0')
    modnet = MODNet().to(device)
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

    _, dataloader = make_train_dataloader(bs)

    start_epoch, current_step = 0, 0
    semantic_loss, detail_loss, matte_loss = 0.0, 0.0, 0.0
    for epoch in range(start_epoch, epochs):
        for idx, batch in enumerate(dataloader):

            current_step += 1

            image, trimap, gt_matte  = batch["image"], batch["trimap"], batch["gt_matte"]
            image = image.type(torch.FloatTensor)
            trimap = trimap.type(torch.FloatTensor)
            gt_matte = gt_matte.type(torch.FloatTensor)

            image = image.to(device)
            trimap = trimap.to(device)
            gt_matte = gt_matte.to(device)

            semantic_loss, detail_loss, matte_loss = \
                supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)

            wandb_dict = {
                "train/global_epoch": epoch + 1,
                "train/global_step": current_step,
                "train/semantic_loss": semantic_loss.item(),
                "train/detail_loss": detail_loss.item(),
                "train/matte_loss": matte_loss.item(),
            }
            wandb.log(wandb_dict)

            print(f"Epoch: {epoch}, Step: {current_step}, LR: {get_lr(optimizer)} " + \
                f"semantic_loss: {round(semantic_loss.item(), 2)}, " + \
                     f"detail_loss: {round(detail_loss.item(), 2)}, matte_loss: {round(matte_loss.item(), 2)}")

        lr_scheduler.step()
        if epoch != 0 and (epoch + 1) % save_feq == 0:
            modnet.eval()
            torch.save(modnet.state_dict(), 
                f"/data/docker/pengyuyan/models/modnet/modnet_epoch{epoch}_step{current_step}_seloss{round(semantic_loss.item(), 2)}_deloss_{round(detail_loss.item(), 2)}_matteloss{round(matte_loss.item(), 2)}.pth")
            modnet.train()
    
if __name__ == "__main__":
    train()