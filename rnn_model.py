from model import Model
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class RNNModelSteering(Model):

    def __init__(self, steer_file):
        self.max = 0
        self.min = 0
        self.log = []
        self.old = 0

        i2h, i2h_b, h2h, h2h_b, h2o, h2o_b = self.load_weights(steer_file)

        self.i2h = Variable(torch.FloatTensor(i2h))
        self.h2h = Variable(torch.FloatTensor(h2h))
        self.h2o = Variable(torch.FloatTensor(h2o))
        self.i2h_b = Variable(torch.FloatTensor(i2h_b))
        self.h2h_b = Variable(torch.FloatTensor(h2h_b))
        self.h2o_b = Variable(torch.FloatTensor(np.array([h2o_b])))

    def predict(self, state):

        data = [state.speed_x, state.distance_from_center, state.angle / 180] + [state.distances_from_edge[2]] + \
            [state.distances_from_edge[8]] + [state.distances_from_edge[9]] + \
            [state.distances_from_edge[10]] + [state.distances_from_edge[16]]

        x_input = Variable(torch.FloatTensor(data))

        output = self.forward(x_input)

        if output[0] > output[1]:
            self.acceleration = output[0] * 10
        else:
            self.breaking = output[1]

        self.acceleration = output[0]
        self.breaking = output[1]
        steer = 0.0

        if output[2] > 0 and output[2] > output[3]:
            steer = -output[2]
            if steer < -1:
                steer = -1
        elif output[3] > 0 and output[3] > output[2]:
            steer = output[3]
            if steer > 1:
                steer = 1
        self.steering = steer

        self.old = self.steering

    def forward(self, x_in):
        hidden = torch.matmul(self.i2h, x_in) + self.i2h_b
        hidden = F.sigmoid(hidden)
        output = torch.matmul(self.h2o, hidden) + self.h2o_b
        return output.data[0]

    def load_weights(self, fn):
        i2h = np.loadtxt(open(fn + 'linear_in_hid.out', 'r'))
        i2h_bias = np.loadtxt(open(fn + 'linear_in_hid_bias.out', 'r'))
        h2h = np.loadtxt(open(fn + 'linear_hid_hid.out', 'r'))
        h2h_bias = np.loadtxt(open(fn + 'linear_hid_hid_bias.out', 'r'))
        h2o = np.loadtxt(open(fn + 'linear_hid_out.out', 'r'))
        h2o_bias = np.loadtxt(open(fn + 'linear_hid_out_bias.out', 'r'))
        return i2h, i2h_bias, h2h, h2h_bias, h2o, h2o_bias
