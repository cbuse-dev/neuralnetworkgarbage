import numpy
import os

class network:
  def __init__(self, inputs, target, neurons, learning_rate=0.001):
    self.x = inputs
    self.target = target
    self.learning_rate = learning_rate
    self.network_architecture = numpy.array([inputs.shape[0], neurons[0], neurons[1]]) #inputs - hidden - outputs

  def clear(self):
    os.system('clear')

  def sigmoid(self,sop):
      return 1.0/(1+numpy.exp(-1*sop))

  def error(self,predicted, target):
      return numpy.power(predicted-target, 2)

  def error_predicted_deriv(self,predicted, target):
      return 2*(predicted-target)

  def sigmoid_sop_deriv(self,sop):
      return self.sigmoid(sop)*(1.0-self.sigmoid(sop))

  def sop_w_deriv(self,x):
      return x

  def update_w(self,w, grad, learning_rate):
      return w - learning_rate*grad

  def train(self):
    x = self.x
    target = self.target

    learning_rate = self.learning_rate

    # Number of inputs, number of neurons per each hidden layer, number of output neurons
    network_architecture = self.network_architecture

    # Initializing the weights of the entire network
    w = []
    w_temp = []
    for layer_counter in numpy.arange(network_architecture.shape[0]-1):
        for neuron_nounter in numpy.arange(network_architecture[layer_counter+1]):
            w_temp.append(numpy.random.rand(network_architecture[layer_counter]))
        w.append(numpy.array(w_temp))
        w_temp = []
    w = numpy.array(w)
    w_old = w

    for iterations,k in enumerate(range(80000)):
        # Forward Pass
        # Hidden Layer Calculations
        sop_hidden = numpy.matmul(w[0], x)

        sig_hidden = self.sigmoid(sop_hidden)

        # Output Layer Calculations
        sop_output = numpy.sum(w[1][0]*sig_hidden)

        predicted = self.sigmoid(sop_output)
        err = self.error(predicted, target)

        # Backward Pass
        g1 = self.error_predicted_deriv(predicted, target)

        ### Working with weights between hidden and output layer
        g2 = self.sigmoid_sop_deriv(sop_output)
        g3 = self.sop_w_deriv(sig_hidden)
        grad_hidden_output = g3*g2*g1
        w[1][0] = self.update_w(w[1][0], grad_hidden_output, learning_rate)
        
        ### Working with weights between input and hidden layer
        g5 = self.sop_w_deriv(x)
        for neuron_idx in numpy.arange(w[0].shape[0]):
            g3 = self.sop_w_deriv(w_old[1][0][neuron_idx])
            g4 = self.sigmoid_sop_deriv(sop_hidden[neuron_idx])
            grad_hidden_input = g5*g4*g3*g2*g1
            w[0][neuron_idx] = self.update_w(w[0][neuron_idx], grad_hidden_input, learning_rate)

        w_old = w
        
        if iterations%5000 == 0:
          pstr = "iteration "+str(iterations)+" - "+str(predicted)
          print("\n{0:.2f}% complete\n".format(100*(iterations/80000))+pstr)
