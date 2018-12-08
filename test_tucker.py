from tensorly.decomposition import parafac, partial_tucker, tucker
import numpy as np
import torch.nn as nn
from numpy.linalg import norm, svd

###############
# Convolution #
###############

# weight = np.random.rand(3,32,5,5)
#
# core, [last, first] = \
#         partial_tucker(weight, \
#             modes=[0, 1], ranks=[3, 10], init='svd')


##########
# Linear #
##########

# weight = nn.Linear(576, 9).weight.data.numpy()
# core, factors = tucker(weight, rank=300)
#
# error = weight - np.dot(factors, core.T)
# loss = norm(error, 'fro')

##############
# Linear SVD #
##############

weight = nn.Linear(576, 9).weight.data.numpy()
U, s, V = np.linalg.svd(weight)