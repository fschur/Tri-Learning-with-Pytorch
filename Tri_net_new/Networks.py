import torch
import torch.nn as nn

# Implement our models here
# Highway is an implementation by: https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py

class Highway(nn.Module):
    def __init__(self, size, num_layers, f, dropout_rate):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear
            x = self.dropout(x)

        return x

