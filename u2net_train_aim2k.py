import glob
import numpy as np
import os
from tqdm import tqdm

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import wandb

from data_loader import ChangeBG
from data_loader import RandomCrop
from data_loader import RandomFlip
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import MatObjDataset
from data_loader import ToTensor
from data_loader import ToTensorLab_Mat

from model import U2NET
from model import U2NETP

from u2net_infer import norm_pred
from u2net_infer import postprocess_composed
import utils.image_utils as image_utils


bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0, labels_v)
	loss1 = bce_loss(d1, labels_v)
	loss2 = bce_loss(d2, labels_v)
	loss3 = bce_loss(d3, labels_v)
	loss4 = bce_loss(d4, labels_v)
	loss5 = bce_loss(d5, labels_v)
	loss6 = bce_loss(d6, labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss0, loss


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



def concat_img(img_list):
    h = max([img.shape[0] for img in img_list]) 
    w = sum([img.shape[1] for img in img_list])
    result_img = np.zeros((h, w, 3))
    start_h, start_w = 0, 0
    for img in img_list:
        result_img[start_h:start_h+img.shape[0], start_w:start_w+img.shape[1], :] = img
        start_w += img.shape[1] 
    return result_img


@torch.no_grad()
def validate(val_loader, net, device, epoch, step):

    stack_images = []
    avg_loss0, avg_total_loss = 0.0, 0.0

    for data in tqdm(val_loader):

        input, label = data['image'], data['label']

        input = input.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor) 

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = input.to(device), label.to(device)

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        avg_loss0 += loss2.item()
        avg_total_loss += loss.item()

        pred = d1[:, 0, :, :]
        pred = norm_pred(pred)
        predict, label = pred.squeeze(), label.squeeze()
        predict_np, label = predict.cpu().data.numpy()[..., np.newaxis], label.cpu().data.numpy()[..., np.newaxis]
        predict_np, label = predict_np * 255, label * 255
        input = image_utils.tensor2uint(input)
        composed = postprocess_composed(predict_np, input).astype(np.uint8)
        stacked_img = concat_img([input, label, predict_np, composed])
        stack_images.append(stacked_img)

    wandb_dict = {
        "val/global_epoch": epoch,
        "val/global_step": step,
        "val/loss0": avg_loss0 / len(val_loader),
        "val/sum_loss": avg_total_loss / len(val_loader),
        "val/vis": wandb.Image(np.vstack(stack_images), caption="Input|Label|Pred|Composed"),
    }
    wandb.log(wandb_dict)
    return avg_loss0 / len(val_loader), avg_total_loss / len(val_loader) 
        

def get_train_list(data_dir, tra_image_dir, tra_label_dir, image_ext='.jpg', label_ext='.png'):
    
    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'+image_ext))
    tra_img_name_list.sort()

    tra_lbl_name_list = glob.glob(os.path.join(data_dir, tra_label_dir, '*'+label_ext)) 
    tra_lbl_name_list.sort()
    return tra_img_name_list, tra_lbl_name_list


def make_train_dataloader(batch_size_train):
    train_datasets = {
        "AIM-train": {
            "data_dir":"/data/docker/pengyuyan/dataset/AIM-2k",
            "tra_image_dir": "train/original",
            "tra_label_dir": "train/mask",
        },
        "AIM-val": {
            "data_dir":"/data/docker/pengyuyan/dataset/AIM-2k",
            "tra_image_dir": "validation/original",
            "tra_label_dir": "validation/mask",
        },
        "google-stray-cats-hd": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/stray_cats_hd",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
        "google-stray-dogs-hd": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/stray_dogs_hd",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
    }

    tra_img_name_list, tra_lbl_name_list = [], []
    for key, value in train_datasets.items():
        data_dir, tra_image_dir, tra_label_dir = value["data_dir"], value["tra_image_dir"], value["tra_label_dir"]
        img_list, label_list = get_train_list(data_dir, tra_image_dir, tra_label_dir)
        tra_img_name_list.extend(img_list)
        tra_lbl_name_list.extend(label_list)
    

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = MatObjDataset(
	img_name_list=tra_img_name_list,
	lbl_name_list=tra_lbl_name_list,
	transform=transforms.Compose([
		ChangeBG(bg_dir="/data/docker/pengyuyan/dataset/google_image_downloader/furiends/bg"),
		RescaleT(320),
		RandomCrop(288),
		RandomFlip(),
		ToTensorLab_Mat()]))

    salobj_dataloader = DataLoader(
        salobj_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, num_workers=16)

    return train_num, salobj_dataloader


# def make_val_dataloader(batch_size_val):
#     data_dir = "/data/docker/pengyuyan/dataset/AIM-2k"
#     val_image_dir = "validation/original"
#     val_label_dir = "validation/mask"
#     image_ext = '.jpg'
#     label_ext = '.png'
    
#     val_img_name_list = glob.glob(os.path.join(data_dir, val_image_dir, '*'+image_ext))
#     val_img_name_list.sort()

#     val_lbl_name_list = glob.glob(os.path.join(data_dir, val_label_dir, '*'+label_ext)) 
#     val_lbl_name_list.sort()

#     print("---")
#     print("val images: ", len(val_img_name_list))
#     print("val labels: ", len(val_lbl_name_list))
#     print("---")

#     val_num = len(val_img_name_list)

#     salobj_dataset = SalObjDataset(
#     img_name_list=val_img_name_list,
#     lbl_name_list=val_lbl_name_list,
#     transform=transforms.Compose([
#         RescaleT(320),
#         ToTensorLab(flag=0)]))

#     salobj_dataloader = DataLoader(
#         salobj_dataset, 
#         batch_size=batch_size_val, 
#         shuffle=False, num_workers=1)

#     return val_num, salobj_dataloader


def train():

    wandb.init(project="furiends", entity="pengyuyan")

    wandb.define_metric("train/global_step")
    # wandb.define_metric("val/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    # wandb.define_metric("val/*", step_metric="val/global_step")

    device = torch.device("cuda:2")

    # ------- 2. set the directory of training dataset --------
    model_name = 'u2net' #'u2netp'
    model_dir = os.path.join(f"/data/docker/pengyuyan/models/{model_name}/0521_exp2_bg_augmented/")
    os.makedirs(model_dir, exist_ok=True)

    epoch_num = 100000
    batch_size_train = 12
    batch_size_val = 1

    train_num, train_salobj_dataloader = make_train_dataloader(batch_size_train)
    # val_num, val_salobj_dataloader = make_val_dataloader(batch_size_val)
    
    # ------- 3. define model --------
    # define the net
    if model_name == 'u2net':
        start_epoch, start_step = 0, 0
        net = U2NET(3, 1)
        net.load_state_dict(torch.load("saved_models/u2net/u2net.pth", map_location=device), strict=True)

        # start_epoch, step, _ = load_checkpoint("/data/docker/pengyuyan/models/u2net/0521_exp1_bg_augmented/u2net_bce_itr_12000_trainloss_0.022_tar_0.219.pth", net, device=device)

    elif(model_name == 'u2netp'):
        net = U2NETP(3,1)

    
    if torch.cuda.is_available():
        net.to(device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    # running_loss = 0.0
    # running_tar_loss = 0.0
    # ite_num4val = 0
    save_frq = 500 # save the model every 2000 iterations
    log_interval = 10

    for epoch in range(start_epoch, epoch_num):
        net.train()

        for i, data in enumerate(train_salobj_dataloader):
            ite_num = ite_num + 1
            # ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['gt_matte']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = inputs.to(device), labels.to(device)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # # print statistics
            # running_loss += loss.item()
            # running_tar_loss += loss2.item()

            wandb_dict = {
                "train/global_epoch": epoch + 1,
                "train/global_step": ite_num,
                "train/loss0": loss2.item(),
                "train/sum_loss": loss.item(),
            }
            wandb.log(wandb_dict)

            if ite_num % log_interval == 0:
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss0: %3f, train loss sum: %3f " % (\
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, \
                        train_num, ite_num, loss2.item(),\
                             loss.item()))

            if ite_num % save_frq == 0:
                net.eval()
                # val_loss0, val_loss_sum = validate(val_salobj_dataloader, net, device, epoch, ite_num)

                # torch.save(
                #     net.state_dict(), 
                #     model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (\
                #         ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

                val_loss0, val_loss_sum = loss2.item(), loss.item()
                save_checkpoint(
                    os.path.join(model_dir, \
                    f"{model_name}_bce_itr_{ite_num}_trainloss_{round(val_loss0, 3)}_tar_{round(val_loss_sum, 3)}.pth"),
                    net, epoch=epoch, step=ite_num)

                # running_loss = 0.0
                # running_tar_loss = 0.0
                net.train()  # resume train
                # ite_num4val = 0
            # net.train()


if __name__ == "__main__":
    train()