import cv2
import glob
import numpy as np
import os

import torch

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from src.models.modnet import MODNet


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name, img_org, pred, d_dir):

    org_shape = img_org.shape[:2]
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = predict_np * 255
    predict_np = cv2.resize(predict_np, (
        org_shape[1], org_shape[0]), interpolation=cv2.INTER_LINEAR)

    composed = postprocess_composed(predict_np, img_org)
    concat_result = np.concatenate([img_org, composed], axis=1)

    img_name = os.path.splitext(os.path.basename(image_name))[0]
    cv2.imwrite(os.path.join(d_dir, img_name+".png"), predict_np)
    # cv2.imwrite(os.path.join(d_dir, "composed_"+img_name+".jpg"), composed)
    cv2.imwrite(os.path.join(d_dir, "concat_"+img_name+".jpg"), concat_result)


def preprocess(image):

    def resize(image, dst_size):
        return cv2.resize(image, dst_size, interpolation=cv2.INTER_LINEAR)


    def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        image = image / np.max(image)
        image = (image - np.array(mean)) / np.array(std)
    
        return image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image, (320, 320))
    image = normalize(image)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return torch.from_numpy(image)


def postprocess_composed(predict, image_org):

    if predict.ndim == 2:
        predict = predict[..., np.newaxis]

    composed = (predict / 255.0) * image_org +\
         (1.0 - predict / 255.0) * np.array([255, 255, 255])
    return composed


def load_model(model_name, model_dir):
    if model_name.split("_")[0] == "u2net":
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name.split("_")[0] == "u2netp":
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    elif model_name.split("_")[0] == "modnet":
        net = MODNet()
    
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(os.path.join(model_dir, model_name))["model_state_dict"])
        # net.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
        net.cuda()
    else:
        net.load_state_dict(torch.load(os.path.join(model_dir, model_name)["model_state_dict"], map_location='cpu'))
        # net.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
    net.eval()

    return net


@torch.no_grad()
def main():

    model_name = "u2net_bce_itr_1000_trainloss_0.201_tar_1.516.pth"  #u2netp
    image_dir = os.path.join(os.getcwd(), 'test_data', 'furiends_test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + f'_results_{image_dir.split("/")[-1]}' + os.sep)
    model_dir = "/data/docker/pengyuyan/models/u2net/0521_exp2_bg_augmented/"

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(len(img_name_list))

    net = load_model(model_name, model_dir)

    for i, img_name in enumerate(img_name_list):
        img_arr = cv2.imread(img_name)
        img_tensor = preprocess(img_arr)
        img_tensor = img_tensor.type(torch.FloatTensor)
        img_tensor = img_tensor.cuda()

        d1 = net(img_tensor)[0]

        pred = d1[:, 0, :, :]
        pred = norm_pred(pred)

        # _, _, matte = net(img_tensor, True)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i], img_arr, pred, prediction_dir)
        # save_output(img_name_list[i], img_arr, matte, prediction_dir)


if __name__ == "__main__":
    main()