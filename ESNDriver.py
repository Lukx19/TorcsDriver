import torch
from torch.autograd import Variable
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from get_FFNdata import read_data, get_batch
import time

# Echo State Network Reservoir module


class Reservoir(nn.Module):
    """

    Echo State Network Reservoir module

    """
    def __init__(self, n_inputs, n_outputs, bsz, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001,
                 teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=lambda x: x, inverse_out_activation=lambda x: x,
                 silent=True, cuda=True):

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
        self.cuda = cuda

        if self.cuda:
            self.states = Variable(torch.zeros(bsz, self.n_reservoir + self.n_inputs).cuda(), requires_grad=False)
            # Initialize reservoir activations
            self.x = Variable(torch.zeros(self.n_reservoir).cuda(), requires_grad=False)
        else:
            self.states = Variable(torch.zeros(bsz, self.n_reservoir + self.n_inputs), requires_grad=False)
            # Initialize reservoir activations
            self.x = Variable(torch.zeros(self.n_reservoir), requires_grad=False)

        # Linear output
        self.lin = nn.Linear(self.n_reservoir + self.n_inputs, self.n_outputs)

    def initReservoirWeights(self):
        # initialize recurrent weights:
        weights = torch.rand(self.n_reservoir, self.n_reservoir) - 0.5
        eigenvals, _ = torch.eig(weights)
        radius = torch.max(torch.abs(eigenvals))
        if self.cuda:
            weights.cuda()
            eigenvals.cuda()
        weights = weights * (self.spectral_radius / radius)
        self.w = Variable(weights, requires_grad=False)

        # random input weights:
        weights_in = torch.rand(self.n_reservoir,
            self.n_inputs) * 2 - 1
        if self.cuda:
            weights_in.cuda()
        self.w_in = Variable(weights_in, requires_grad=False)
        # random feedback output weights:
        weights_feedb = torch.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1
        if self.cuda:
            weights_feedb.cuda()
        self.w_feedb = Variable(weights_feedb, requires_grad=False)
        weights_out = torch.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1
        if self.cuda:
            weights_out.cuda()
        self.w_out = Variable(weights_out, requires_grad=False)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def calcReservoir(self, state, inpt, outp):
        # if self.teacher_forcing:
        #     preactivation = ((self.w @ state) + (self.w_in @ inpt) + (self.w_feedb @ outp))
        # else:
        #     preactivation = ((self.w @ state)
        #                      + torch.dot(self.w_in @ inpt))
        preactivation = ((self.w @ state) + (self.w_in @ inpt) + (self.w_feedb @ outp))
        noise_tensor = torch.rand(self.n_reservoir) - 0.5
        # if self.cuda:
        #     noise_tensor.cuda()
        noise = Variable(self.noise * noise_tensor, requires_grad=False)
        return torch.tanh(preactivation) + noise

    def forward(self, inputs, outputs, init_state=None):
        # Batch size
        batch_size = inputs.size()[0]

        # States
        self.states.data.zero_()

        # to get rid of memory issues
        self.states = self.repackage_hidden(self.states)
        self.x = self.repackage_hidden(self.x)

        # if init_state is not None:
        #     self.x = init_state

        state_idx = (0, self.n_reservoir)
        input_idx = (state_idx[1], state_idx[1] + self.n_inputs)
        output_idx = (input_idx[1], input_idx[1] + self.n_outputs)

        for n in range(1, batch_size):
            self.x = self.calcReservoir(self.x, inputs[n, :], outputs[n - 1, :])
            self.states[n, state_idx[0]:state_idx[1]] = self.x
            self.states[n, input_idx[0]:input_idx[1]] = inputs[n, :]
            # states[n, output_idx[0]:output_idx[1]] = outputs[n - 1, :]

        out = self.lin(self.states)

        return out


def evaluate_echo(model, x_valid, y_valid, bsz, loss_function):

    total_loss = 0
    for i in range(len(y_valid) // bsz):
        x_in, target = get_batch(x_valid, y_valid, bsz, i)
        outputs = model(x_in.view(bsz, -1), target.view(bsz, -1))
        loss = loss_function(outputs.view(BATCH_SIZE, -1), target)
        total_loss += bsz * loss.data

    return total_loss / len(y_valid)


def save_weights(dir, model):
    np.savetxt(dir + '/echo_w.out', model.w.data.cpu().numpy())
    np.savetxt(dir + '/echo_w_in.out', model.w_in.data.cpu().numpy())
    np.savetxt(dir + '/echo_w_feedback.out', model.w_feedb.data.cpu().numpy())
    np.savetxt(dir + '/echo_w_out.out', model.w_out.data.cpu().numpy())
    np.savetxt(dir + '/echo_lin.out', model.lin.weight.data.cpu().numpy())
    np.savetxt(dir + '/echo_lin_bias.out', list(model.lin.parameters())[1].data.cpu().numpy())


EPOCHS = 30
SPECTRAL_RADIUS = 0.3  # 0.95, 0.6, 0.3
N_RES = 600   # 200, 600, 1000
DATA_FILE = "data/tracks/"
CUDA = False
LEARNING_RATE = 0.01
PRINT_EVERY = 100
SAVEDIR = "EchoWeights/echonn_res600"
# get training set
data = read_data(DATA_FILE, CUDA)
x_train = data["train"]["x"]
y_train = data["train"]["y"]
n_train = len(x_train)
x_valid = data["valid"]["x"]
y_valid = data["valid"]["y"]
n_valid = len(x_valid)
x_test = data["test"]["x"]
y_test = data["test"]["y"]
n_test = len(x_test)
print("Data loaded")
BATCH_SIZE = 100
n_inputs = 8
n_outputs = 4


print('input neurons:', n_inputs, 'output neurons:', n_outputs)
model = Reservoir(n_inputs, n_outputs, BATCH_SIZE, n_reservoir=N_RES, spectral_radius=SPECTRAL_RADIUS, cuda=CUDA)
loss_function = nn.MSELoss()


def train():

    total_loss = 0
    start_time = time.time()

    for i in range(len(y_train) // BATCH_SIZE):
        input_var, target_var = get_batch(x_train, y_train, BATCH_SIZE, i)

        model.zero_grad()
        outputs = model(input_var.view(BATCH_SIZE, -1), target_var.view(BATCH_SIZE, -1))
        loss = loss_function(outputs.view(BATCH_SIZE, -1), target_var)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-LEARNING_RATE, p.grad.data)

        # print(loss)

        total_loss += loss.data

        if i % PRINT_EVERY == 0 and i > 0:
            cur_loss = total_loss[0] / PRINT_EVERY
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} '.format(
                epoch, i, n_train // BATCH_SIZE, LEARNING_RATE,
                              elapsed * 1000 / PRINT_EVERY, cur_loss))
            total_loss = 0
            start_time = time.time()


# initialize best validation loss
best_val_loss = None

# hit Ctrl + C to break out of training early
try:

    # loop over epochs
    for epoch in range(1, EPOCHS):

        epoch_start_time = time.time()
        model.train()
        train()
        val_loss = evaluate_echo(model, x_valid, y_valid, BATCH_SIZE, loss_function)
        val_loss = val_loss.numpy()[0]
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} '.format(
            epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or abs(best_val_loss - val_loss) > 0.0005:
            # save_weights(SAVEDIR, model)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            LEARNING_RATE = LEARNING_RATE / 2
    train_loss = evaluate_echo(model, x_train, y_train, BATCH_SIZE, loss_function)
    valid_loss = evaluate_echo(model, x_valid, y_valid, BATCH_SIZE, loss_function)
    test_loss = evaluate_echo(model, x_test, y_valid, BATCH_SIZE, loss_function)
    print('train loss: ', train_loss.numpy()[0])
    print('valid loss: ', valid_loss.numpy()[0])
    print('test loss: ', test_loss.numpy()[0])

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    print('Calculating losses')
    train_loss = evaluate_echo(model, x_train, y_train, BATCH_SIZE, loss_function)
    valid_loss = evaluate_echo(model, x_valid, y_valid, BATCH_SIZE, loss_function)
    test_loss = evaluate_echo(model, x_test, y_valid, BATCH_SIZE, loss_function)
    print('train loss: ', train_loss.numpy()[0])
    print('valid loss: ', valid_loss.numpy()[0])
    print('test loss: ', test_loss.numpy()[0])
