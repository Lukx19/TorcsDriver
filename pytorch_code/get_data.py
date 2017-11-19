import numpy as np
import random
import torch
from torch.autograd import Variable

def read_data(file, n_targets, dim, acc):

    data = np.loadtxt(open(file, 'r'), delimiter=',', skiprows=1)    
    n_rows = len(data[0])
    target = []
    x_input = []
    if not acc:
        target_index = 1
    else:
        target_index = 0
    for row in data:
        input_temp = row[n_rows - dim:]
        target.append([int(row[target_index])])
        x_input.append([input_temp])

    # split the data into training, test and validation set
    data_length = len(x_input)
    batch_size = 10
    num_batches = data_length / batch_size
    n_train = round(0.8 * data_length)
    n_valid = round(0.1 * data_length)
    n_test = round(0.1 * data_length)

    x_train, x_valid, x_test = torch.FloatTensor(x_input[:n_train]), torch.FloatTensor(x_input[n_train:(n_train+n_valid)]), torch.FloatTensor(x_input[(n_train+n_valid):])
    y_train, y_valid, y_test = torch.FloatTensor(target[:n_train]), torch.FloatTensor(target[n_train:(n_train+n_valid)]), torch.FloatTensor(target[(n_train+n_valid):])
    
    print("n_train: ", n_train)
    print("n_train+n_valid: ", n_train+n_valid)
    print()
    print("%", " of 1s in y_train: ", sum(y_train.numpy())/n_train)
    print("%", ' of 1s in y_valid: ', sum(y_valid.numpy())/n_valid)
    print("%", ' of 1s in y_test: ', sum(y_test.numpy())/n_test)
    print()

    return data_length, n_train, n_valid, n_test, x_train, x_valid, x_test, y_train, y_valid, y_test

def randomTrainingPair(x, y, n):
    x_tensor = []
    y_tensor = []
    rand_index = random.randint(0, x.shape[0] - n)
    for i in range(n):
        x_tensor.append(Variable(x[rand_index + i], requires_grad = True))
        y_tensor.append(Variable((y[rand_index + i])))
    return y_tensor, x_tensor


def sequentialPair(x, y, n, epoch):
    x_tensor = []
    y_tensor = []
    rounds = x.shape[0] // n
    for i in range(n):
        x_tensor.append(Variable(x[(epoch % rounds) * n + i], requires_grad = True))
        y_tensor.append(Variable((y[(epoch % rounds) * n + i])))
    return y_tensor, x_tensor

