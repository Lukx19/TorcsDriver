import numpy as np
import torch
from model_tanh import RNN
import random
from torch.autograd import Variable
import torch.nn as nn
from get_data_steer import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# data pars CHANGE
dimension_x = 22
num_classes = 1
print_every = 5000
loss_every = 1000
epochs = 100000
nn_filename = "steer_norm_aalborg_provided_batch-50.pt"
data_file = "training_data/norm_aalborg_provided.csv"

# get data
data_length, n_train, n_valid, n_test, x_train, x_valid, x_test, y_train, y_valid, y_test = read_data(data_file, num_classes, dimension_x)

# hyperparameters CHANGE
hidden_size = 50
learning_rate = 0.001
batch_length = 50

# set RNN
rnn = RNN(dimension_x, hidden_size, num_classes)

# init loss
total_loss = 0
all_losses = []
mean_squared = nn.MSELoss()

# train
for i in range(epochs):
   
    category_tensor, input_tensor = sequentialPair(x_train, y_train, batch_length, i)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for j in range(len(input_tensor)):
        output, hidden = rnn(input_tensor[j], hidden)
    
    loss = mean_squared(output.view(1,), category_tensor[batch_length - 1])
    loss.backward()
    total_loss += loss.data[0]

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    
    if i % print_every == 0:
        print("Progress: ", str(int(i / epochs * 100)), "%")
        print("Current loss")	
        print(loss.data[0])

    total_loss += loss
    
    # Add current loss avg to list of losses
    if i % loss_every == 0:
        all_losses.append(total_loss.data[0] / loss_every)
        total_loss = 0

torch.save(rnn, nn_filename)
plt.figure()
plt.plot(all_losses[1:])
plt.show()


