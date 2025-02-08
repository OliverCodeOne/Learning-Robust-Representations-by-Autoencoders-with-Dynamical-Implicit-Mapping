import torch.nn as nn
import torch

# nmODE-AE
from nets.NMODEModel import nmODE
from nets.NMODEModel import ODEBlock
class nmODEAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, tol=1e-3, adjoint=False, eval_times = (0, 1)):
        super(nmODEAutoencoder, self).__init__()
        self.eval_times = torch.tensor(eval_times).float()

        # encoder
        # self.encoder1 = nn.Linear(input_size, int(input_size/2))
        self.encoder1 = nn.Linear(input_size, hidden_size)
        self.gelu = nn.GELU()
        self.nmODE_down1 = nmODE()
        self.ode_down1 = ODEBlock(self.nmODE_down1, tol=tol, adjoint=adjoint)
        # self.encoder2 = nn.Linear(int(input_size/2), hidden_size)

        # decoder
        # self.decoder1 = nn.Linear(hidden_size, int(input_size/2))
        self.decoder1 = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.leakyreLU = nn.LeakyReLU()
        # self.nmODE_down2 = nmODE()
        # self.ode_down2 = ODEBlock(self.nmODE_down2, tol=tol, adjoint=adjoint)
        # self.decoder2 = nn.Linear(int(input_size/2), input_size)
        # self.sparsity_weight = sparsity_weight

    def forward(self, x):
        # encode
        x = self.encoder1(x)
        # x = self.gelu(x)
        self.nmODE_down1.fresh(x)
        hidden = self.ode_down1(torch.zeros_like(x), self.eval_times)

        # decode
        x = self.decoder1(hidden)
        # x = self.decoder2(x)
        output = self.sigmoid(x)
        # self.nmODE_down2.fresh(x)
        # output = self.ode_down2(torch.zeros_like(x), self.eval_times)
        return output, hidden