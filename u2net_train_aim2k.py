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

from data_loader import ChangeBGWithDiverseScaleAndPosition
from data_loader import ReplaceBG
from data_loader import RandomCrop
from data_loader import RandomFlip
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import MatObjDataset
from data_loader import ToTensor
from data_loader import ToTensorLab_Mat

from model import U2NET
from model import U2NETP

from infer import norm_pred
from infer import postprocess_composed
import utils.image_utils as image_utils
from utils.misc import get_learning_rate
from utils.misc import load_checkpoint
from utils.misc import save_checkpoint
from utils.misc import concat_img
from pytorch_msssim import msssim


bce_loss = nn.BCELoss(size_average=True)
mssim_loss = msssim

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    ssim_weight = 1.0

    loss0 = bce_loss(d0, labels_v) + ssim_weight * (1 - mssim_loss(d0, labels_v, normalize="relu"))
    loss1 = bce_loss(d1, labels_v) + ssim_weight * (1 - mssim_loss(d1, labels_v, normalize="relu"))
    loss2 = bce_loss(d2, labels_v) + ssim_weight * (1 - mssim_loss(d2, labels_v, normalize="relu"))
    loss3 = bce_loss(d3, labels_v) + ssim_weight * (1 - mssim_loss(d3, labels_v, normalize="relu"))
    loss4 = bce_loss(d4, labels_v) + ssim_weight * (1 - mssim_loss(d4, labels_v, normalize="relu"))
    loss5 = bce_loss(d5, labels_v) + ssim_weight * (1 - mssim_loss(d5, labels_v, normalize="relu"))
    loss6 = bce_loss(d6, labels_v) + ssim_weight * (1 - mssim_loss(d6, labels_v, normalize="relu"))

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


@torch.no_grad()
def validate(val_loader, net, device, epoch, step):

    stack_images = []
    avg_loss0, avg_total_loss = 0.0, 0.0

    for data in tqdm(val_loader):

        image, mask = data['image'], data['gt_matte']

        image = image.type(torch.FloatTensor)
        mask = mask.type(torch.FloatTensor) 

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = image.to(device), mask.to(device)

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        avg_loss0 += loss2.item()
        avg_total_loss += loss.item()

        pred = d1[:, 0, :, :]
        pred = norm_pred(pred)
        predict, mask = pred.squeeze(), mask.squeeze()
        predict_np, mask = predict.cpu().data.numpy()[..., np.newaxis], mask.cpu().data.numpy()[..., np.newaxis]
        image = image_utils.tensor2uint(image)
        composed = postprocess_composed(predict_np, image).astype(np.uint8)
        predict_np, mask = predict_np * 255, mask * 255
        stacked_img = concat_img([image, mask, predict_np, composed])
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
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/stray_dogs_hd_formatted/",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
        "google-stray-cats-hd": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/stray_cats_hd_formatted/",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
        "google-stray-dogs-hd2": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/stray_dogs_hd2_formatted/",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
        "google-stray-dogs-hd2": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/stray_cats_hd2_formatted/",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
        "overhead-shot-dog": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/overhead_shot_dog_formatted/",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
        "overhead-shot-cat": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/overhead_shot_cat_formatted",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        },
        "kitten": {
            "data_dir": "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/kitten_formated",
            "tra_image_dir": "images",
            "tra_label_dir": "masks",
        }
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
		# ChangeBG(bg_dir="/data/docker/pengyuyan/dataset/google_image_downloader/furiends/bg"),
		RescaleT(320),
		RandomCrop(288),
		RandomFlip(),
		ToTensorLab_Mat()]))

    salobj_dataloader = DataLoader(
        salobj_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, num_workers=16)

    return train_num, salobj_dataloader


def make_val_dataloader(batch_size_val):
    data_dir = "/data/docker/pengyuyan/dataset/google_image_downloader/furiends/validation/furiends_test1"
    val_image_dir = "images"
    val_label_dir = "masks"
    image_ext = '.jpg'
    label_ext = '.png'
    
    val_img_name_list = glob.glob(os.path.join(data_dir, val_image_dir, '*'+image_ext))
    val_img_name_list.sort()

    val_lbl_name_list = glob.glob(os.path.join(data_dir, val_label_dir, '*'+label_ext)) 
    val_lbl_name_list.sort()

    print("---")
    print("val images: ", len(val_img_name_list))
    print("val labels: ", len(val_lbl_name_list))
    print("---")

    val_num = len(val_img_name_list)

    salobj_dataset = MatObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        ToTensorLab_Mat()]))

    salobj_dataloader = DataLoader(
        salobj_dataset, 
        batch_size=batch_size_val, 
        shuffle=False, num_workers=1)

    return val_num, salobj_dataloader


def train():

    wandb.init(project="furiends", entity="pengyuyan")

    wandb.define_metric("train/global_step")
    # wandb.define_metric("val/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    # wandb.define_metric("val/*", step_metric="val/global_step")

    device = torch.device("cuda:2")

    # ------- 2. set the directory of training dataset --------
    model_name = 'u2net' #'u2netp'
    model_dir = os.path.join(f"/data/docker/pengyuyan/models/{model_name}/0528_exp1_ssim/")
    os.makedirs(model_dir, exist_ok=True)

    epoch_num = 100000
    batch_size_train = 12
    batch_size_val = 1

    train_num, train_salobj_dataloader = make_train_dataloader(batch_size_train)
    val_num, val_salobj_dataloader = make_val_dataloader(batch_size_val)
    
    # ------- 3. define model --------
    # define the net
    if model_name == 'u2net':
        start_epoch, start_step = 0, 0
        net = U2NET(3, 1)
        # net.load_state_dict(torch.load("saved_models/u2net/u2net.pth", map_location=device), strict=True)

        start_epoch, step, _ = load_checkpoint("/data/docker/pengyuyan/models/u2net/0528_exp1_ssim/u2net_epoch34_bce_itr_6500_trainloss_0.101_tar_0.886.pth", net, device=device)

    elif(model_name == 'u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.to(device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50], gamma=0.1)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    save_frq = 500 # save the model every 2000 iterations
    log_interval = 10

    for epoch in range(start_epoch, epoch_num):
        net.train()

        for i, data in enumerate(train_salobj_dataloader):
            ite_num = ite_num + 1

            images, masks = data['image'], data['gt_matte']

            images = images.type(torch.FloatTensor)
            masks = masks.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v, labels_v = images.to(device), masks.to(device)

            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            wandb_dict = {
                "train/global_epoch": epoch + 1,
                "train/global_step": ite_num,
                "train/loss0": loss2.item(),
                "train/sum_loss": loss.item(),
                "train/lr": get_learning_rate(optimizer),
            }
            wandb.log(wandb_dict)

            if ite_num % log_interval == 0:
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss0: %3f, train loss sum: %3f " % (\
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, \
                        train_num, ite_num, loss2.item(),\
                             loss.item()))

            if ite_num % save_frq == 0:
                net.eval()
                val_loss0, val_loss_sum = validate(val_salobj_dataloader, net, device, epoch, ite_num)

                # val_loss0, val_loss_sum = loss2.item(), loss.item()
                save_checkpoint(
                    os.path.join(model_dir, \
                    f"{model_name}_epoch{epoch}_bce_itr_{ite_num}_trainloss_{round(val_loss0, 3)}_tar_{round(val_loss_sum, 3)}.pth"),
                    net, epoch=epoch, step=ite_num)

                net.train()  # resume train
        # scheduler.step()


if __name__ == "__main__":
    train()