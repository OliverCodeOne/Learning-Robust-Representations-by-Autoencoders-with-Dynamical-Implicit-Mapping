import os
import pdb
import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import torch.nn.functional as F


MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Utility class that wraps odeint and odeint_adjoint.

        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float()
        else:
            integration_time = eval_times.type_as(x)
            
        if self.adjoint:  # self.odefunc是微分方程、x是初始函数值、integration_time为想要计算的时间点
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            # print(out)
            return out[1]

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, non_linearity='relu'):
        """
        Block for ConvODEUNet

        Args:
            in_channels (int): number of filters for the conv layers
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvBlock, self).__init__()

        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.non_linearity(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        return out
    
def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()
    
class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)

class SinSquare(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        return torch.sin(x) * torch.sin(x)

class nmODE(nn.Module):
    def __init__(self):
        """
        """
        super(nmODE, self).__init__()
        self.nfe = 0  # Number of function evaluations
        self.gamma = None
    
    def fresh(self, gamma):
        self.gamma = gamma
    
    def forward(self, t, p):
        self.nfe += 1
        # dpdt = -p + torch.relu(p + self.gamma)
        # dpdt = -p + torch.sin(p + self.gamma) * torch.sin(p + self.gamma)
        # print("----迭代-----")
        dpdt = -p + torch.sigmoid(p + self.gamma)
        return dpdt

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(156, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 30),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 156),
            # nn.ReLU()
        )

    def forward(self, x):
        # encode
        x = self.encoder(x)
        # decode
        x = self.decoder(x)
        return x
    
class nmODEModel(nn.Module):
    def __init__(self, in_channels, output_dim=1, time_dependent=False,
                 non_linearity='softplus', tol=1e-3, adjoint=False, eval_times = (0, 1)):
        """
        ConvODEUNet (U-Node in paper)

        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
        """
        super(nmODEModel, self).__init__()
        # nf = num_filters
        self.eval_times = torch.tensor(eval_times).float()

        # self.linear1 = nn.Linear(in_channels, int(in_channels/2))
        self.linear1 = nn.Linear(in_channels, int(in_channels/2))
        # self.ae = AE()
        self.dropout = nn.Dropout(p=0.2)  # 添加Dropout层
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

        self.nmODE_down1 = nmODE()
        self.ode_down1 = ODEBlock(self.nmODE_down1, tol=tol, adjoint=adjoint)

        self.linear2 = nn.Linear(int(in_channels/2), int(in_channels/4))

        # self.nmODE_down2 = nmODE()
        # self.ode_down2 = ODEBlock(self.nmODE_down2, tol=tol, adjoint=adjoint)
        # self.norm = nn.LazyBatchNorm1d()
        self.classifier = nn.Linear(int(in_channels/4), output_dim)

        # self.fc1 = nn.Linear(7, 120)
        # self.fc2 = nn.Linear(120, 60)
        # self.fc4 = nn.Linear(60, 2)
        
        # self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x, return_features=False):
        x = self.linear1(x)
        # x = self.ae.encoder(x)
        x = self.gelu(x)
        self.nmODE_down1.fresh(x)
        x = self.ode_down1(torch.zeros_like(x), self.eval_times)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        pred = self.classifier(x)
        # self.nmODE_down1.fresh(x)
        # x = self.ode_down1(torch.zeros_like(x), self.eval_times)
        # x = F.relu(self.fc2(x))
        # x = self.fc4(x)

        return pred