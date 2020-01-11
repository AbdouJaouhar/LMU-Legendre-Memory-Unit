import torch.nn as nn
from .LMU import *


class LMUModel(nn.Module):
  def __init__(self):
    super(LMUModel, self).__init__()
    self.LMU = LegendreMemoryUnit(1,49,4,4)
    self.dense = nn.Linear(49,1)

  def forward(self,x):
    x, _ = self.LMU(x)
    x = self.dense(x)

    return x

class LSTMModel(nn.Module):
  def __init__(self):
    super(LSTMModel, self).__init__()
    self.LSTM = nn.LSTM(1,25,1,batch_first=True)
    self.dense = nn.Linear(25,1)

  def forward(self,x):
    x, _ = self.LSTM(x)
    x = self.dense(x)

    return x
