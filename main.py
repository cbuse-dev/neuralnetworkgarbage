import numpy as np
import neuralnetwork as nn

net = nn.network(np.array([0.4, 0.6, 2.0]), np.array(0.2), [8,1])
net.train()
