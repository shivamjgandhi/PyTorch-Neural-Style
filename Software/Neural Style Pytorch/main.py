from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import numpy as np 

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Import the style and content images

imsize = 512 if torch.cuda.is_available() else 128

# this turns the data into a tensor and normalizes
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def image_loader(img_name):
	image = Image.open(img_name)
	image = transform(image).unsqueeze(0)
	return image.to(device, torch.float)

style_img = image_loader()
content_img = image_loader()

assert style_img.size() == content_img.size(), \
	"Two images aren't the same size"

# create a function that displays an image by reconverting a copy
# of it to PIL format and siplaying the copy using plt.imshow

unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	image = unloader(image)
	plt.imshow(image)
	if title is None:
		plt.title("image")
	plt.pause(0.001)

plt.figure()
imshow(style_img)

plt.figure()
imshow(content_img)