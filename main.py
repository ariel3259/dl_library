import numpy as np

from network import Network
from functions.activation_fn import tanth, tanth_prime
from functions.loss_fn import mse, mse_prime
from layers.activation_layer import ActivationLayer
from layers.fc_layer import FCLayer


#training data
x_train = np.array([[[0,0]], [[0,1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])