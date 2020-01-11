from tqdm import tqdm
import torch.utils.data as data
from .models import *
from .utils import *

(train_X, train_Y), (test_X, test_Y) = generate_data(128, 5000)

print("Training of LMU")

model = LMUModel().cuda()
print("\n\nNombre de paramètres : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n\n")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

dataset = torch.utils.data.TensorDataset(torch.Tensor(train_X).cuda(), torch.Tensor(train_Y).cuda())
dataset = data.DataLoader(dataset, batch_size=16, shuffle=True)

dataset_valid = torch.utils.data.TensorDataset(torch.Tensor(test_X).cuda(), torch.Tensor(test_Y).cuda())
dataset_valid = data.DataLoader(dataset_valid, batch_size=16, shuffle=False)

train(model, 100, dataset, dataset_valid)


print("Training of LSTM")

model = LSTMModel().cuda()
print("\n\nNombre de paramètres : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("\n\n")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

dataset = torch.utils.data.TensorDataset(torch.Tensor(train_X).cuda(), torch.Tensor(train_Y).cuda())
dataset = data.DataLoader(dataset, batch_size=16, shuffle=True)

dataset_valid = torch.utils.data.TensorDataset(torch.Tensor(test_X).cuda(), torch.Tensor(test_Y).cuda())
dataset_valid = data.DataLoader(dataset_valid, batch_size=16, shuffle=False)

train(model, 100, dataset, dataset_valid)
