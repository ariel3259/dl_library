import numpy as np

from network import Network
from functions.activation_fn import tanth, tanth_prime
from functions.loss_fn import mse, mse_prime
from layers.activation_layer import ActivationLayer
from layers.fc_layer import FCLayer


#training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()

#network
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanth, tanth_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanth, tanth_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs = 100, learning_rate=0.1)

#test
out = net.predict(x_train)
print(out)