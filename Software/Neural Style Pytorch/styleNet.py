import torch
import torch.nn as nn
import torch.nn.functional as F 

import torchvision.models as models
import copy 

from loss_functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# VGG networks are trained on images with each channel normalized by
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in
# a nn.Sequential

class normalize(nn.Module):
	def __init__(self, mean, std):
		super(normalize, self).__init__()
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, img):
		return (img - self.mean)/self.std

# based on the paper, we can see which layers are default usage for content and style
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_loss(cnn, normalization_mean, normalization_std, 
	style_img, content_img, content_layers = content_layers_default,
	style_layers = style_layers_default):

	cnn = copy.deepcopy(cnn)

	# normalization module
	normalization = normalize(normalization_mean, normalization_std).to(device)

	# to have iterable access to list of losses
	content_losses = []
	style_losses = []

	# We create a new sequential model to put in modules that
	# are activated sequentially
	model = nn.Sequential(normalization)

	# want to iterate for every convolution layer
	i = 0
	for layer in cnn.children():
		if isinstance(layer, nn.Conv2d):
			i = i+1
			name = 'conv_{}'.format(i)
		elif isinstance(layer, nn.ReLU):
			name = 'relu_{}'.format(i)
			layer = nn.ReLU(inplace = False)
		elif isinstance(layer, nn.MaxPool2d):
			name = 'pool2d_{}'.format(i)
		elif isinstance(layer, nn.BatchNorm2d):
			name = 'batchNorm_{}'.format(i)
		else:
			raise RunTimeError("Uncrecognized layer {}".format(layer.__class__.__name__))

		model.add_module(name, layer)

		if name in content_layers:
			# Compute the content loss
			FCL = model(content_img).detach()
			content_loss = contentLoss(FCL)
			model.add_module("content_loss_{}".format(i), content_loss)
			content_losses.append(content_loss)
		elif name in style_layers:
			# Compute the style loss
			FSL = model(style_img).detach()
			style_loss = styleLoss(FSL)
			model.add_module("style_loss_{}".format(i), style_loss)
			style_losses.append(style_loss)

	# trim off the layers after the last content and style losses
	for i in range(len(model) - 1, -1, -1):
		if isinstance(model[i], contentLoss) or isinstance(model[i], styleLoss):
			break

	model = model[:(i+1)]

	return content_losses, style_losses, model

