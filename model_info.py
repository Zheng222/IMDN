import torch

model1 = torch.load('checkpoints/IMDN_AS.pth')
model2 = torch.load('checkpoint_x2/epoch_250.pth')

print(model1)