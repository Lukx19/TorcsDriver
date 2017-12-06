from model import Model
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
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

            # i2h, i2h_b, h2h, h2h_b, h2o, h2o_b = self.load_weights(acc_file)
            #
            # self.i2h_acc = Variable(torch.cuda.FloatTensor(i2h).cpu())
            # self.h2h_acc = Variable(torch.cuda.FloatTensor(h2h).cpu())
            # self.h2o_acc = Variable(torch.cuda.FloatTensor(h2o).cpu())
            #
            # self.i2h_b_acc = Variable(torch.cuda.FloatTensor(i2h_b).cpu())
            # self.h2h_b_acc = Variable(torch.cuda.FloatTensor(h2h_b).cpu())
            # self.h2o_b_acc = Variable(torch.cuda.FloatTensor(np.array([h2o_b])).cpu())

        # else:
        #     self.model = torch.load(steer_file)
        #     self.hidden = self.model.init_hidden()

    def predict(self, state):

        # for nn with new data from teachers
        # data = [state.speed_x, state.speed_y, state.speed_z, state.distance_from_center, state_copy.angle / 180] + [state_copy.distances_from_edge[0]] + [state_copy.distances_from_edge[5]] + [state_copy.distances_from_edge[13]] + [state_copy.distances_from_edge[18]]

        # for nn with 4 sensors
        # data = [state.speed_x, state.distance_from_center, state.angle / 180] + [state.distances_from_edge[0]] + [
        #    state.distances_from_edge[5]] + [state.distances_from_edge[13]] + [state.distances_from_edge[18]]

        # for nn with 4 front sensors
        data = [state.speed_x, state.distance_from_center, state.angle / 180] + [state.distances_from_edge[2]] + [state.distances_from_edge[8]] + [state.distances_from_edge[9]] + [state.distances_from_edge[10]] + [state.distances_from_edge[16]]

        # for nn with 2 sensors
        # data =  [state_copy.speed_x, state_copy.distance_from_center, state_copy.angle / 180] + [state_copy.distances_from_edge[0]] + [state_copy.distances_from_edge[18]]

        # for nn with 0 sensors
        # data =  [state_copy.speed_x, state_copy.distance_from_center, state_copy.angle / 180]

        # for nn with all sensors
        # data = [state.speed_x, state.angle / 180, state.distance_from_center] + list(state.distances_from_edge)



        x_input = Variable(torch.FloatTensor(data))
        output = self.forward(x_input)
        # self.steering = output[2]
        # self.steering = 0.01 * output[2] + 0.99 * self.old
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
        # print('acc')
        # print(self.acceleration)
        # print('brake')
        # print(self.breaking)
        # print('steer')
        # print(output[2])
        # print(output[3])
        # print(self.steering)
        # if abs(state.distance_from_center) <= 1:
        #     # prediction = output
        #     self.steering = 0.01 * output[2] + 0.99 * self.old
        # else:
        #     # prediction = output
        #     # prediction = self.old + 0.01 * output
        #     self.steering = 0.2 * output + 0.8 * (0.0 - state.distance_from_center)
        # if abs(state.distance_from_center) <= 1:
        #     # prediction = output
        #     prediction = 0.01 * output + 0.99 * self.old
        # else:
        #     prediction = 0.2 * output + 0.8 * (0.0 - state.distance_from_center)
        # self.steering = prediction
        self.old = self.steering
        # else:
        #
        #     x_input = Variable(torch.FloatTensor(data))
        #     output, self.hidden = self.model(x_input.view(-1, 1, len(data)), self.hidden)
        #     print(output.data[0].cpu().numpy()[0])
        #     prediction = self.old * 0.99 + output.data[0].cpu().numpy()[0][0] * 0.01
        #     self.steering = prediction
        #     self.old = prediction

    def forward(self, x_in):
        hidden = torch.matmul(self.i2h, x_in) + self.i2h_b
        hidden = F.sigmoid(hidden)
        # hidden = torch.matmul(self.h2h, hidden) + self.h2h_b
        # hidden = F.tanh(hidden)
        output = torch.matmul(self.h2o, hidden) + self.h2o_b
        # output = F.sigmoid(output)
        return output.data[0]

    # def forward_acc(self, x_in):
    #     hidden = torch.matmul(self.i2h_acc, x_in) + self.i2h_b_acc
    #     hidden = F.tanh(hidden)
    #     hidden = torch.matmul(self.h2h_acc, hidden) + self.h2h_b_acc
    #     hidden = F.tanh(hidden)
    #     output = torch.matmul(self.h2o_acc, hidden) + self.h2o_b_acc
    #     return output.data[0]

    def load_weights(self, fn):

        i2h = np.loadtxt(open(fn + 'linear_in_hid.out', 'r'))
        i2h_bias = np.loadtxt(open(fn + 'linear_in_hid_bias.out', 'r'))

        h2h = np.loadtxt(open(fn + 'linear_hid_hid.out', 'r'))
        h2h_bias = np.loadtxt(open(fn + 'linear_hid_hid_bias.out', 'r'))

        h2o = np.loadtxt(open(fn + 'linear_hid_out.out', 'r'))
        h2o_bias = np.loadtxt(open(fn + 'linear_hid_out_bias.out', 'r'))

        return i2h, i2h_bias, h2h, h2h_bias, h2o, h2o_bias
