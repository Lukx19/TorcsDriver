import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.m = nn.Dropout(p=0.1)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.outlayer = nn.Linear(hidden_size, output_size)

    def forward(self, x_input, hidden):
        hidden = self.layer1(x_input) + self.layer2(hidden)
        hidden = F.tanh(hidden)
        hidden = self.m(hidden)
        output = self.outlayer(hidden)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

