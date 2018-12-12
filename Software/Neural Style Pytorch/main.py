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
import cv2

import copy

from loss_functions import *
from gradient_descent import *
from styleNet import *

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

style_img = image_loader('picasso.jpg')
content_img = image_loader('dancing.jpg')

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
	plt.title("image")
	plt.pause(0.1)

plt.figure()
imshow(style_img)

plt.figure()
imshow(content_img)

input_img = content_img.clone()

plt.figure()
imshow(input_img)

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

output = run_style_transfer(cnn, cnn_norm_mean, cnn_norm_std,
	content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()