import torch
from torch import nn
from .ODEBlock import ODEBlock


class NMODEFunc(nn.Module):
    def __init__(self, in_features, out_features, T=0.2, non_linearity='SinSquare', time_dependent=False):
        super(NMODEFunc, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.T = T
        # self.w = nn.Sequential(
        #     nn.Linear(in_features, out_features),
        # )
        self.w = nn.Sequential(
            nn.Linear(in_features, out_features),
            # nn.Sigmoid()
            # nn.BatchNorm1d(44),
            # nn.LeakyReLU(),
            # nn.Linear(44, out_features),

            # nn.Conv1d(1, 16, 2),  
            # nn.ReLU(),
            # nn.MaxPool1d(2),  
            # # nn.Conv1d(16, 32, 2),
            # # nn.ReLU(),
            # # nn.MaxPool1d(4),  # 输出大小：torch.Size([128, 32, 1])
            # nn.Flatten(),  # 输出大小：torch.Size([128, 32])
            # nn.Linear(688, out_features),
        )

        print("in_features:" + str(in_features))
        print("out_features:" + str(out_features))
        self.out_features = out_features

        self.non_linearity = get_nonlinearity(non_linearity)
        self.wx = None
        pass

    def pre_forward(self, x):
        self.nfe = 0
        self.wx = self.w(x)
        # print("pre_forward")

    def init_hidden(self, x):
        return torch.zeros(x.shape[0], self.out_features).type_as(x)

    def get_integration_time(self):
        if self.training:
            t = self.T + (torch.rand(1) - 0.5) * 0.6 * self.T
        else:
            t = self.T
        return torch.tensor([0., t]).float()

    def forward(self, t, y):
        # print("forward")
        # self.nfe += 1
        # y = -y + self.non_linearity(self.wx + y)
        return y


class K2NMODEFunc(NMODEFunc):
    def __init__(self, in_features, out_features, T=1, non_linearity='SinSquare', time_dependent=False):
        super().__init__(in_features, out_features, T, non_linearity, time_dependent)

    def forward(self, t, y):
        self.nfe += 1
        y = -y + self.non_linearity(torch.cos(self.wx + y) + y)
        return y


def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name.lower() == 'relu':
        return nn.ReLU(inplace=True)
    if name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'softplus':
        return nn.Softplus()
    elif name.lower() == 'lrelu':
        return nn.LeakyReLU()
    elif name.lower() == 'sinsquare':
        return SinSquare()
    elif name.lower() == 'swish':
        return SinSquare()


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SinSquare(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        return torch.sin(x) * torch.sin(x)


class NMODENet(nn.Module):
    def __init__(self, conf, num_class=10, **kwargs):
        super(NMODENet, self).__init__()
        self.model_name = conf.get("model_name", "")
        non_linearity = conf["non_linearity"]
        num_hidden = [conf["in_features"]] + conf["num_hidden"]
        # print(f"num_hidden[-1]:{num_hidden[-1]}, num_class:{num_class}")
        self.net = nn.Sequential(*[ODEBlock(
            odefunc=globals()[conf["model_name"]](num_hidden[i], num_hidden[i + 1], non_linearity=non_linearity[i])) for
            i in range(len(num_hidden) - 1)])
        # self.net = nn.Sequential(
        #     nn.Linear(88, 40),
        #     # nn.Sigmoid(),
        #     nn.Linear(40, 2),
        #     # nn.Sigmoid(),
        # )
        print(self.net)
        self.fc = nn.Linear(num_hidden[-1], num_class)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x, require_y=False):
        # print("net*******")
        out = self.net(x)
        out = self.fc(out)
        # out = self.relu(out)
        # out = self.sigmoid(out)
        # print("out*******")
        # print(out.shape)
        return out
