import numpy as np
import random
import torch
import csv
from torch.autograd import Variable
import os
from random import shuffle


def read_data(file, cuda):

    target = []
    x_input = []
    target_test = []
    target_valid =[]
    x_input_test = []
    x_input_valid = []

    for fn in os.listdir(file):
        train_temp_input = []
        train_temp_target = []
        valid_temp_input = []
        valid_temp_target = []
        test_temp_input = []
        test_temp_target = []
        with open(file + fn, 'r') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rows = sum(1 for row in rowreader)
        with open(file + fn, 'r') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            counter = 0
            for row in rowreader:
                # input_temp = [float(i) for i in row[3:6]] + [float(row[6]), float(row[11]), float(row[19]), float(row[24])]
                #input_temp = [float(i) for i in row[3:6]] + [float(row[6]), float(row[9]), float(row[11]), float(row[16]), float(row[19]),
                #                                             float(row[24])]
                #input_temp = [float(i) for i in row[3:6]] + [float(row[6]), float(row[24])]
                # input_temp = [float(i) for i in row[3:6]] + [float(row[6]), float(row[24])]
                input_temp = [float(i) for i in row[3:6]] + [float(row[7])] + [float(i) for i in row[14:17]] + [float(row[23])]
                # input_temp = [float(i) for i in row[3:6]] + [float(row[8])] + [float(row[14])] + [float(row[16])] + [
                #     float(row[22])]
                # input_temp = [float(i) for i in row[3:]]
                #input_temp = [float(i) for i in row[3:6]]

                right_steer = 0
                left_steer = 0
                if float(row[2]) > 0:
                    right_steer = float(row[2])
                elif float(row[2]) < 0:
                    left_steer = abs(float(row[2]))

                if counter < 141:
                    train_temp_target.append([float(i) for i in row[0:2]] + [left_steer, right_steer])
                    train_temp_input += [input_temp]
                    counter += 1
                elif counter < 171:
                    valid_temp_target.append([float(i) for i in row[0:2]] + [left_steer, right_steer])
                    valid_temp_input += [input_temp]
                    counter += 1
                elif counter < 201:
                    test_temp_target.append([float(i) for i in row[0:2]] + [left_steer, right_steer])
                    test_temp_input += [input_temp]
                    counter += 1
                else:
                    counter = 0
            x_input += train_temp_input
            target += train_temp_target
            target_test += test_temp_target
            x_input_test += test_temp_input
            target_valid += valid_temp_target
            x_input_valid += valid_temp_input

    print('Creating tensors, can take a while ...')
    if cuda:
        x_train = torch.cuda.FloatTensor(x_input)
        y_train = torch.cuda.FloatTensor(target)
        x_valid = torch.cuda.FloatTensor(x_input_valid)
        y_valid = torch.cuda.FloatTensor(target_valid)
        x_test = torch.cuda.FloatTensor(x_input_test)
        y_test = torch.cuda.FloatTensor(target_test)
    else:
        x_train = torch.FloatTensor(x_input)
        y_train = torch.FloatTensor(target)
        x_valid = torch.FloatTensor(x_input_valid)
        y_valid = torch.FloatTensor(target_valid)
        x_test = torch.FloatTensor(x_input_test)
        y_test = torch.FloatTensor(target_test)

    return {
        "train": {
            "x": x_train,
            "y": y_train
        },
        "valid": {
            "x": x_valid,
            "y": y_valid
        },
        "test": {
            "x": x_test,
            "y": y_test
        }
    }


def randomTrainingPair(x, y, n):  
    print("\n type x: \n", type(x))
    print("\n type y: \n", type(y))
    print("length x: ", len(x))
    print("\n shape x: \n", x.size())
    print("\n shape y: \n", y.size())
    x_tensor = torch.FloatTensor(1,22).zero_()
    y_tensor = torch.LongTensor([0]) #(1,1)
    rand_index = random.randint(0, len(x) - n)
    for i in range(n):
        # print("\n x randindex: \n", x[rand_index+i])
        # print("\n y randindex: \n", y[rand_index+i])
        # print("\n y randindex 1 1: \n", y[rand_index+i].view(1,1))
        x_tensor = torch.cat((x_tensor,x[rand_index+i]),0)
        y_tensor = torch.cat((y_tensor,y[rand_index+i].view(1,1)),0)
        # x_tensor.append(x[rand_index+i])
        # y_tensor.append(y[rand_index+i])
    # print("x_tensor type: \n", type(x_tensor))
    print("\n type x_tensor: \n", type(x_tensor))
    print("\n x_tensor: \n", x_tensor)
    # x_tensor = torch.LongTensor()
    print("\n type y_tensor: \n", type(y_tensor))
    print("\n y_tensor: \n", y_tensor)
    # y_tensor = torch.LongTensor()
    return x_tensor, y_tensor


def get_batch(x, y, bsz, j):
    rounds = x.shape[0] // bsz
    batch_x = x[(j % rounds) * bsz + 0:(j % rounds) * bsz + bsz]
    batch_y = y[(j % rounds) * bsz + 0:(j % rounds) * bsz + bsz]
    return Variable(batch_x), Variable(batch_y)
