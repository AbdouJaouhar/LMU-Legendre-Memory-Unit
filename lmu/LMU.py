import torch
import torch.nn as nn
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
import numpy as np
from functools import partial
import torch.nn.functional as F
import math

from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay
from functools import partial

'''
Initialisation LECUN_UNIFOR
- tensor to fill
- fan_in is the input dimension size
'''
def lecun_uniform(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))


class LegendreMemoryUnitCell(nn.Module):
    """
    Cell encoding Legendre Memory Unit.

    Parameters
    ----------
    input_dim : int
        Dimension size of input sequence
    order : int
        Order of the Legendre expansion
    theta : positive float
        length of window

    [input/hidden/memory]_[kernel/encoders]_initializer : Fn
        Function used to initialize weight tensors
    Returns
    -------
    int
        Description of return value

    """
  def __init__(self, input_dim, units , order, theta,
                 input_encoders_initializer=lecun_uniform,
                 hidden_encoders_initializer=lecun_uniform,
                 memory_encoders_initializer=partial(torch.nn.init.constant_, val=0),
                 input_kernel_initializer=torch.nn.init.xavier_normal_,
                 hidden_kernel_initializer=torch.nn.init.xavier_normal_,
                 memory_kernel_initializer=torch.nn.init.xavier_normal_):
    super(LegendreMemoryUnitCell, self).__init__()

    self.order = order
    self.theta = theta
    self.units = units


    realizer = Identity()
    self._realizer_result = realizer(LegendreDelay(theta=theta, order=self.order))

    self._ss = cont2discrete(self._realizer_result.realization, dt=1., method='zoh')

    self._A = self._ss.A - np.eye(order)
    self._B = self._ss.B
    self._C = self._ss.C

    self.AT = nn.Parameter(torch.Tensor(self._A), requires_grad=False)
    self.BT = nn.Parameter(torch.Tensor(self._B), requires_grad=False)


    self.encoder_input = nn.Parameter(torch.Tensor(1,input_dim), requires_grad=True)
    self.encoder_hidden = nn.Parameter(torch.Tensor(1,self.units), requires_grad=True)
    self.encoder_memory = nn.Parameter(torch.Tensor(1,self.order ), requires_grad=True)
    self.kernel_input = nn.Parameter(torch.Tensor(self.units, input_dim), requires_grad=True)
    self.kernel_hidden = nn.Parameter(torch.Tensor(self.units, self.units), requires_grad=True)
    self.kernel_memory = nn.Parameter(torch.Tensor(self.units, self.order), requires_grad=True)


    input_encoders_initializer(self.encoder_input)
    hidden_encoders_initializer(self.encoder_hidden)
    memory_encoders_initializer(self.encoder_memory)
    input_kernel_initializer(self.kernel_input)
    hidden_kernel_initializer(self.kernel_hidden)
    memory_kernel_initializer(self.kernel_memory)

  def EulerOdeSolver(self):

    """
    Simple Euler solver of the ordinary differential equation (ODE) :

        theta * m'(t) = A*m(t) + B*u(t)

    m : d-dimensional state-vector
    u : input signal

    (A, B) : state space. Here A and B are initialized using Padé approximant (generated thanks author's paper code)

    """
    A_hat = (self.step_delta_t/self.theta)*self.AT + torch.eye(self.order,self.d_order_ode)
    B_hat = (self.step_delta_t/self.theta)*self.BT

    return A_hat, B_hat

  def forward(self, xt, states):

    """
    Forward call of the LMU Cell

    Parameters
    ----------
    x_{t} : tensor
        tensor representing a signal or feature signal
    states : tuple of tensors
        tuple containing previouses h_{t} and m_{t} states vectors

    Returns
    -------
    x_{t+1}, (h_{t+1}, m_{t+1})

    """
    ht, mt = states

    ut = F.linear(xt, self.encoder_input) + F.linear(ht, self.encoder_hidden) + F.linear(mt, self.encoder_memory)

    mt = mt + F.linear(mt, self.AT) + F.linear(ut, self.BT)

    ht = nn.Tanh()(F.linear(xt, self.kernel_input) + F.linear(ht, self.kernel_hidden) + F.linear(mt, self.kernel_memory))

    return ht, (ht, mt)



class LegendreMemoryUnit(nn.Module):
    """
    Implementation of LMU using LegendreMemoryUnitCell so it can be used as LSTM or GRU in PyTorch Implementation (no GPU acceleration)
    """
  def __init__(self, input_dim, units , order, theta):
    super(LegendreMemoryUnit, self).__init__()

    self.units = units
    self.order = order

    self.lmucell = LegendreMemoryUnitCell(input_dim, units , order, theta)

  def forward(self, xt):
    outputs = []

    h0 = torch.zeros(xt.size(0),self.units).cuda()
    m0 = torch.zeros(xt.size(0),self.order).cuda()
    states = (h0,m0)
    for i in range(xt.size(1)):
      out, states = self.lmucell(xt[:,i,:], states)
      outputs += [out]
    return torch.stack(outputs).permute(1,0,2), states
