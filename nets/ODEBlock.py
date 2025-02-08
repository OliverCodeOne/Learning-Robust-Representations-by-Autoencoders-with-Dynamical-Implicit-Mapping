from torchdiffeq import odeint, odeint_adjoint
from torch import nn

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol
        pass

    def forward(self, x, eval_times=None):
        # print("----------BLOCK-FORWARD------")
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.pre_forward(x)
        if eval_times is None:
            integration_time = self.odefunc.get_integration_time().type_as(x) #TODO: TRICK time in range [0, 1+-0.1]
        else:
            print(eval_times)
            integration_time = eval_times.type_as(x)
        # set gamma as initial value for y_0
        y_0 = self.odefunc.init_hidden(x)
        if self.adjoint:
            out = odeint_adjoint(self.odefunc, y_0, integration_time,
                                rtol=self.tol, atol=self.tol, method='dopri5',
                                options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, y_0, integration_time,
                        rtol=self.tol, atol=self.tol, method='dopri5',
                        options={'max_num_steps': MAX_NUM_STEPS})
        return out[-1]
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        pass