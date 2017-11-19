import numpy as np
import torch
from model import RNN
import random
from torch.autograd import Variable
import torch.nn as nn
from get_data import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.optim as optim

if __name__ == "__main__":

    # data pars CHANGE
    dimension_x = 22
    num_classes = 1
    print_every = 1000
    loss_every = 1000
    epochs = 100000
    nn_filename = "class-sequence-50-acc_norm_Forza_1_41_50_full_model_hidden_50.pt"
    data_file = "training_data/norm_Forza_1_41_50.csv"
    
    # true for accelaration as target, false for break
    acc = True

    # get data
    data_length, n_train, n_valid, n_test, x_train, x_valid, x_test, y_train, y_valid, y_test = read_data(data_file, num_classes, dimension_x, acc)

    # hyperparameters CHANGE
    hidden_size = 50
    learning_rate = 0.005
    batch_length = 50

    # set RNN
    rnn = RNN(dimension_x, hidden_size, num_classes)

    # init loss
    total_loss = 0
    all_losses = []
    cross_entropy = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    
    # train
    for i in range(epochs):
       
        #category_tensor, input_tensor = randomTrainingPair(x_train, y_train, batch_length)
        category_tensor, input_tensor = randomTrainingPair(x_train, y_train, batch_length)
        hidden = rnn.initHidden()
    
        rnn.zero_grad()
    
        for j in range(len(input_tensor)):
            output, hidden = rnn(input_tensor[j], hidden)
    
        loss = cross_entropy(output.view(1,), category_tensor[batch_length - 1])
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
    
        # Add parameters' gradients to their values, multiplied by learning rate
        #for p in rnn.parameters():
        #    p.data.add_(-learning_rate, p.grad.data)
    
        if i % print_every == 0:
            print("Progress: ", str(int(i / epochs * 100)), "%")
            if i > loss_every:
                print("Current loss")
                print(all_losses[-1])
    
        total_loss += loss
        
        # Add current loss avg to list of losses
        if i % loss_every == 0:
            all_losses.append(total_loss.data[0] / loss_every)
            total_loss = 0

    # save neural net
    torch.save(rnn, nn_filename)
    plt.figure()
    plt.plot(all_losses[1:])
    plt.show()


