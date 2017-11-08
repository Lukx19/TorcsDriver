import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# Echo State Network Reservoir module

class Reservoir(nn.Module):
    """

    Echo State Network Reservoir module

    """
    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001,
                 teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=lambda x: x, inverse_out_activation=lambda x: x,
                 silent=True):

        """
        Args:

            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            silent: supress messages

        """
        super(Reservoir, self).__init__()

        # check for proper dimensionality of all arguments and write them down.

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.teacher_forcing = teacher_forcing
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.initReservoirWeights()

        # Initialize reservoir activations
        self.x = Variable(torch.zeros(self.n_reservoir), requires_grad=False)
        # Linear output
        self.out = nn.Linear(self.n_reservoir, self.n_outputs)

    def initReservoirWeights(self):
        # initialize recurrent weights:
        weights = torch.rand(self.n_reservoir, self.n_reservoir) - 0.5
        radius = torch.max(torch.abs(torch.eigvals(weights)))
        weights = weights * (self.spectral_radius / radius)
        self.w = Variable(weights, requires_grad=False)

        # random input weights:
        self.w_in = Variable(torch.rand(self.n_reservoir,
            self.n_inputs) * 2 - 1,requires_grad=False)
        # random feedback output weights:
        self.w_feedb = Variable(torch.rand(self.n_reservoir,
            self.n_outputs) * 2 - 1,requires_grad=False)

    def calcReservoir(self,state,inpt,outp):
        if self.teacher_forcing:
            preactivation = (torch.dot(self.w, state)
                             + torch.dot(self.w_in, inpt)
                             + torch.dot(self.w_feedb, outp))
        else:
            preactivation = (torch.dot(self.w, state)
                             + torch.dot(self.w_in, inpt))
        return (torch.tanh(preactivation)
                + self.noise * (torch.rand(self.n_reservoir) - 0.5))
    # Forward
    def forward(self, inputs,outputs,init_state=None):
        # Batch size
        batch_size = inputs.size()[0]

        # States
        states = Variable(torch.zeros(batch_size, self.n_reservoir), requires_grad=False)
        if init_state is not None:
            states[0,:] = init_state

        for n in range(1,batch_size):
            states[n, :] = self.calcReservoir(states[n-1,:],inputs[n,:],outputs[n-1,:])
            # states[n, :] = x

        # Linear output
        out_activation = self.out(states[-1,:])
        return out_activation, states[-1,:]

    def predict(self,inputs):
        # Batch size
        batch_size = inputs.size()[0]

        states = torch.zeros(batch_size, self.n_reservoir)
        outputs = torch.zeros(batch_size, self.n_outputs)
        states[0,:] = self.calcReservoir(states[0,:],inputs[0,:],torch.zeros(1, self.n_reservoir))
        for n in range(1,batch_size):
            states[n, :] = self.calcReservoir(states[n-1,:],inputs[n,:],outputs[n-1,:])
            outputs[n,:] = self.out(states[n, :])
        return outputs[-1,:]

# end Reservoir

def trainESN(input_batches,target_batches, reservoir_size =20):
    net = Reservoir(n_inputs = 5, n_outputs = 3,n_reservoir=500,spectral_radius=0.95)
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for i in range(len(input_batches)):
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output,_ = net(input_batches[i])
        criterion = nn.MSELoss()
        loss = criterion(output, target_batches[i])
        loss.backward()
        optimizer.step()
    return net
