import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super(FFN, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_in_hid = nn.Linear(input_size, hidden_size)
        self.linear_hid_hid = nn.Linear(hidden_size, hidden_size)
        self.linear_hid_out = nn.Linear(hidden_size, output_size)

    def forward(self, x_input):
        hidden = self.linear_in_hid(x_input)
        hidden = F.tanh(hidden)
        dropped = self.drop(hidden)
        hidden = self.linear_hid_hid(dropped)
        hidden = self.drop(hidden)
        output = self.linear_hid_out(hidden)
        return output
