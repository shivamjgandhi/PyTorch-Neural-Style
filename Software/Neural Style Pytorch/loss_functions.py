import torch
import torch.nn as nn
import torch.nn.functional as F 

class contentLoss(nn.Module):
	"""
	This is the class for computing contentLoss.

	The loss is mean-squared error between the content image and the 
	output of the layer at layer L.
	"""
	def __init__(self, FCL):
		super(contentLoss, self).__init__()
		self.FCL = FCL.detach()

	def forward(self, FXC):
		self.loss = F.mse_loss(self.FCL, FXC)
		return FXC

def gramMatrix(FXL):
	"""
	Computes the gram matrix of the matrix FXL.
	"""
	a, b, c, d = FXL.size()
	# a = batch size
	# b = number of feature maps
	# c, d = size of matrix itself
	hatFXL = FXL.view(a*b, c*d)
	gram = torch.mm(hatFXL, torch.t(hatFXL))
	return gram.div(a*b*c*d)

class styleLoss(nn.Module):
	"""
	This is the class for computing styleLoss.

	Looks almost exactly like the loss for the content.
	"""
	def __init__(self, FSL):
		super(styleLoss, self).__init__()
		self.GSL = gramMatrix(FSL).detach()

	def forward(self, FXL):
		GXL = gramMatrix(FXL)
		self.loss = F.mse_loss(self.GSL, GXL)
		return FXL