# data loader
from __future__ import print_function, division
import cv2
import glob
import os
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import torch

import utils.file_utils as file_utils


#==========================dataset load==========================
class ReplaceBG(object):
	def __init__(self, bg_dir, prob=0.5):
		self.bg_list = file_utils.get_files_recursively(bg_dir)
		self.prob = prob
	
	def __call__(self, sample):
		if random.random() > self.prob:
			return sample

		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		bg_path = random.choice(self.bg_list)
		bg_arr = cv2.imread(bg_path)
		bg_arr = cv2.resize(bg_arr, (image.shape[1], image.shape[0]))

		composed = (label / 255.0) * image + (1.0 - label / 255.0) * bg_arr
		sample["image"] = composed

		return sample


class ChangeBGWithDiverseScaleAndPosition(object):
	def __init__(self, bg_dir, prob=0.5):
		self.bg_list = file_utils.get_files_recursively(bg_dir)
		self.prob = prob
	
	def __call__(self, sample):
		if random.random() > self.prob:
			return sample

		_, image, label = sample['imidx'], sample['image'], sample['label']
		bg_path = random.choice(self.bg_list)
		bg_arr = cv2.imread(bg_path)
		img_h, img_w = image.shape[:2]
		bg_h, bg_w = bg_arr.shape[:2]

		# 如果背景的最长边比图片的最长边要大，那就讲图片resize到比背景小的程度或者浮动
		if max(bg_h, bg_w) > max(img_h, img_w):
			base_scale = max(bg_h/ img_h, bg_w/img_w)
			scale_factor = random.choice([i / 10.0 for i in range(int(base_scale * 5), int(base_scale * 10))])
			image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor)
			label = cv2.resize(label, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)[..., np.newaxis]
		else:
			base_scale = max(img_h/bg_h, img_w/bg_w)
			scale_factor = random.choice([i / 10.0 for i in range(int(base_scale * 10),  int(base_scale * 20))])
			bg_arr = cv2.resize(bg_arr, dsize=None, fx=scale_factor, fy=scale_factor)
		
		img_h, img_w = image.shape[:2]
		bg_h, bg_w = bg_arr.shape[:2]
		
		bg_pil = Image.fromarray(bg_arr)
		label_pil = Image.fromarray(np.zeros(bg_arr.shape[:2], dtype=np.uint8))
		img_pil = Image.fromarray(image)
		
		top_left = (random.randint(0, max((bg_w-img_w)//2, 1)), random.randint(0, max((bg_h-img_h)//2, 1)))
		bg_pil.paste(img_pil, top_left, Image.fromarray(label[...,0]))
		label_pil.paste(Image.fromarray(label[..., 0]), top_left, Image.fromarray(label[...,0]))

		sample['image'] = np.array(bg_pil).astype(np.uint8)
		sample["label"] = np.array(label_pil).astype(np.uint8)[..., np.newaxis]

		return sample


class RandomFlip(object):
	def __init__(self, hflip=True, rot=True) -> None:
		self.hflip = hflip
		self.rot = rot

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		hflip = self.hflip and random.random() < 0.5
		vflip = self.rot and random.random() < 0.5
		rot90 = self.rot and random.random() < 0.5

		def _augment(img):
			if hflip: img = img[:, ::-1, :] 
			if vflip: img = img[::-1, :, :]
			if rot90: img = img.transpose(1, 0, 2)
			
			return img

		sample['image'] = _augment(image)
		sample['label'] = _augment(label)

		return sample
		
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}


class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}


class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]


		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ToTensorLab_Mat(object):
	def __call__(self, sample):
		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
		image = image / 255.0
		if image.shape[2] == 1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225	
		
		tmpLbl[:,:,0] = label[:,:,0]
		tmpTrimap = tmpLbl.copy()
		tmpTrimap[(0 < tmpTrimap) * (tmpTrimap < 1)] = 0.5
		
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))
		tmpTrimap = tmpTrimap.transpose((2, 0, 1))

		return {
			'imidx':torch.from_numpy(imidx), 
			'image': torch.from_numpy(tmpImg), 
			'trimap': torch.from_numpy(tmpTrimap), 
			'gt_matte': torch.from_numpy(tmpLbl.copy())
		}


class SalObjDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list, transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image = io.imread(self.image_name_list[idx])
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample


class MatObjDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list, transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = cv2.imread(self.image_name_list[idx])
		imidx = np.array([idx])

		if 0 == len(self.label_name_list):
			label = np.zeros(image.shape)
		else:
			label = cv2.imread(self.label_name_list[idx], 0)[..., np.newaxis]

		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample

def check_dataset():
	import utils.image_utils as image_utils
	import shutil 

	if os.path.exists("./augment"):
		shutil.rmtree("./augment")

	os.makedirs("./augment", exist_ok=True)
	data_dir = "/data/docker/pengyuyan/dataset/AIM-2k"
	tra_image_dir = "train/original"
	tra_label_dir = "train/mask"
	image_ext = '.jpg'
	label_ext = '.png'

	tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'+image_ext))
	tra_img_name_list.sort()

	tra_lbl_name_list = glob.glob(os.path.join(data_dir, tra_label_dir, '*'+label_ext)) 
	tra_lbl_name_list.sort()

	salobj_dataset = MatObjDataset(
	img_name_list=tra_img_name_list,
	lbl_name_list=tra_lbl_name_list,
	transform=transforms.Compose([
		ChangeBGWithDiverseScaleAndPosition(bg_dir="/data/docker/pengyuyan/dataset/google_image_downloader/furiends/bg", prob=1.0),
		RescaleT(640),		
		RandomCrop(576),
		RandomFlip(),
		ToTensorLab_Mat()]))

	NUM_SAMPLES = 10
	for i, data in enumerate(salobj_dataset):
		input = data["image"]
		trimap = data["trimap"]
		gt_matte = data["gt_matte"]
		img = image_utils.tensor2uint(input)
		trimap = image_utils.tensor2uint_label(trimap)
		gt_matte = image_utils.tensor2uint_label(gt_matte)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(f"./augment/org_{i}.jpg", img)
		cv2.imwrite(f"./augment/trimap_{i}.png", trimap)
		cv2.imwrite(f"./augment/gt_matte_{i}.png", gt_matte)
		if i >= NUM_SAMPLES:
			break
	

if __name__ == "__main__":
	check_dataset()