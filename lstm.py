import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bsz):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.m = nn.Dropout(p=0.4)
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.4)
        self.hidden2output = nn.Linear(hidden_size, output_size)
        self.bsz = bsz

    def forward(self, x_in, hidden):
        lstm_out, self.hidden = self.lstm(x_in, hidden)
        # lstm_out = self.m(lstm_out)
        output = self.hidden2output(lstm_out)
        return output, self.hidden

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.hidden_size).cuda()),
                Variable(torch.zeros(1, 1, self.hidden_size)).cuda())
