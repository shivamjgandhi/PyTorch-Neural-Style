import torch
import torch.nn as nn
import torch.nn.functional as F 

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