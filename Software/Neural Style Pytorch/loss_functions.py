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
		super(StyleLoss, self).__init__()
		self.FCL = gram_matrix(FCL).detach()

	def forward(self, FXC):
		loss = nn.MSELoss(self.FCL, FXC)
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
	gram = hatFXL

class styleLoss(nn.Module):
	"""