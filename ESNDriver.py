import torch
from torch.autograd import Variable
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

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
        # self.lin = nn.Linear(self.n_reservoir, self.n_outputs)

    def initReservoirWeights(self):
        # initialize recurrent weights:
        weights = torch.rand(self.n_reservoir, self.n_reservoir) - 0.5
        eigenvals,_ = torch.eig(weights)
        radius = torch.max(torch.abs(eigenvals))
        weights = weights * (self.spectral_radius / radius)
        self.w = Variable(weights,requires_grad = False)

        # random input weights:
        self.w_in = Variable(torch.rand(self.n_reservoir,
            self.n_inputs) * 2 - 1,requires_grad = False)
        # random feedback output weights:
        self.w_feedb = Variable(torch.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1,requires_grad = False)

        self.w_out = Variable(torch.rand(
            self.n_reservoir,self.n_outputs) * 2 - 1,requires_grad = False)

        # self.w = self.w.type(torch.DoubleTensor)
        # self.w_in = self.w_in.type(torch.DoubleTensor)
        # self.w_feedb = self.w_feedb.type(torch.DoubleTensor)


    def calcReservoir(self,state,inpt,outp):
        if self.teacher_forcing:
            preactivation = ((self.w @ state) + (self.w_in @ inpt)+ (self.w_feedb @ outp))
        else:
            preactivation = ((self.w @ state)
                             + torch.dot(self.w_in @ inpt))
        noise = Variable(self.noise * (torch.rand(self.n_reservoir) - 0.5) ,requires_grad=False)
        return (torch.tanh(preactivation)
                + noise)
    # Forward
    def forward(self, inputs,outputs,init_state=None):
        # Batch size
        batch_size = inputs.size()[0]

        # States
        states = Variable(torch.zeros(batch_size,
            self.n_reservoir + self.n_inputs + self.n_outputs),requires_grad=False)

        if init_state is not None:
            states[0,:] = init_state
        state_idx = (0,self.n_reservoir)
        input_idx = (state_idx,state_idx[1]+self.n_inputs)
        output_idx = (input_idx[1],input_idx[1]+self.n_outputs)

        for n in range(1,batch_size):
            self.x = self.calcReservoir(states[n-1,:],inputs[n,:],outputs[n-1,:])
            states[n, state_idx[0]:state_idx[1]] = self.x
            states[n, input_idx[0]:input_idx[1]] = inputs[n,:]
            states[n, output_idx[0]:output_idx[1]] = outputs[n-1,:]

        out_activation = self.x * self.w_out
        return out_activation, states[-1,:]
    def fit(self,M,T):
        self.w_out = Variable(torch.from_numpy(
            np.dot(np.linalg.pinv(M.data.numpy()),np.arctanh(targets)).T))


    def predict(self,inputs):
        # Batch size
        batch_size = inputs.size()[0]

        states = torch.zeros(batch_size, self.n_reservoir)
        outputs = torch.zeros(batch_size, self.n_outputs)
        states[0,:] = self.calcReservoir(states[0,:],inputs[0,:],torch.zeros(1, self.n_reservoir))
        for n in range(1,batch_size):
            states[n, :] = self.calcReservoir(states[n-1,:],inputs[n,:],outputs[n-1,:])
            outputs[n,:] = states[n, :] * self.w_out
        return outputs[-1,:]

# end Reservoir

def trainESN(dataset,batch_size,n_inputs,n_outputs,reservoir_size =20,):
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=1)
    net = Reservoir(n_inputs = n_inputs, n_outputs = n_outputs,n_reservoir=500,spectral_radius=0.95)
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for num,data in enumerate(batches):
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        inputs = Variable(data[0])
        targets = Variable(data[1])
        output,M = net.forward(inputs,targets)
        net.fit(M,targets)

        criterion = nn.MSELoss()
        print(output)
        output = net.predict(inputs)
        loss = criterion(output, targets)
        print(loss.data)
    return net


data = pd.read_csv("data/alpine-1.csv",sep=',')
# targets = data[['ACCELERATION','BRAKE','STEERING']]
targets = data[['BRAKE']]
inputs = data.loc[:,'SPEED':'ANGLE_TO_TRACK_AXIS']
batch_size = 100
inputs = copy.deepcopy(torch.from_numpy(inputs.as_matrix()).type(torch.FloatTensor))
targets = copy.deepcopy(torch.from_numpy(targets.as_matrix()).type(torch.FloatTensor))
dataset = TensorDataset(inputs, targets)
n_inputs = inputs.shape[1]
n_outputs = targets.shape[1]
print ('input neurons:',n_inputs,'output neurons:',n_outputs)
net = trainESN(dataset = dataset,batch_size = 30, reservoir_size =500, n_inputs=n_inputs,n_outputs=n_outputs)