import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = 'cuda'

class SAE(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.enc = nn.Linear(n, m, bias=True)
        self.dec = nn.Linear(m, n, bias=True)
        nn.init.zeros_(self.enc.bias)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        f = F.relu(self.enc(x))
        out = self.dec(f)
        return f, out
    
device = 'cuda'
act_4conv2 = torch.from_numpy(np.load('act_4conv2.npy'))
act_2conv1 = torch.from_numpy(np.load('act_2conv1.npy'))

class ConvActDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# sample a few neurons in the 128 28x28 feature maps (conv layer2) and in the 512 7x7 feature maps (conv layer4)
# just pick first and last channels and neurons
# get the activations from all images
batch_size = 128

act4conv2c0n0 = act_4conv2[:, :, 0, 0, 0].view(-1, 1)
act4conv2c0n48 = act_4conv2[:, :, 0, -1, -1].view(-1, 1)
act4conv2c511n0 = act_4conv2[:, :, -1, 0, 0].view(-1, 1)
act4conv2c511n48 = act_4conv2[:, :, -1, -1, -1].view(-1, 1)
act4_data = torch.stack([act4conv2c0n0, act4conv2c0n48, act4conv2c511n0, act4conv2c511n48], dim=1).squeeze()

act2conv1c0n0 = act_2conv1[:, :, 0, 0, 0].view(-1, 1)
act2conv1c0n783 = act_2conv1[:, :, 0, -1, -1].view(-1, 1)
act2conv1c127n0 = act_2conv1[:, :, -1, 0, 0].view(-1, 1)
act2conv1c127n783 = act_2conv1[:, :, -1, -1, -1].view(-1, 1)
act2_data = torch.stack([act2conv1c0n0, act2conv1c0n783, act2conv1c127n0, act2conv1c127n783], dim=1).squeeze()

act4_dataset = ConvActDataset(act4_data)
act2_dataset = ConvActDataset(act2_data)
act4_dl = torch.utils.data.DataLoader(act4_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
act2_dl = torch.utils.data.DataLoader(act2_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# train sae with acts from resnet hooks
act_dl = act4_dl
run_name = 'act4'
d_act = next(iter(act_dl)).shape[-1]
d_feat = 100
model = SAE(d_act, d_feat)
model = model.to(device)
lam = 5

optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=5e-5)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

losses = []
model.train()
optimizer.zero_grad()
epochs = 10

for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for orig_acts in act_dl:
        orig_acts = orig_acts.to(device)
        feats, recon_acts = model(orig_acts)
        l2 = torch.sum(torch.square(orig_acts - recon_acts), dim=-1)
        l1 = torch.zeros(feats.shape[0]).to(device)
        for i in range(feats.shape[1]):
            l1 += (torch.abs(feats[:, i]) * torch.linalg.norm(model.dec.weight[:, i]))
        l1 = lam * l1
        loss = torch.mean(l2 + l1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        losses.append(loss.item())
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")
    # checkpoint(model, epoch, optimizer, loss.item(), run_name)

torch.save(model, f"{run_name}.pth")

losses = np.array(losses)
np.save(f'loss_{run_name}', losses)

plt.figure()
plt.plot(losses)
plt.title(f'loss: sae layer 4 conv2')
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig(f'loss_sae4conv2.pdf')

# train sae with acts from resnet hooks
act_dl = act2_dl
run_name = 'act2'
d_act = next(iter(act_dl)).shape[-1]
d_feat = 100
model = SAE(d_act, d_feat)
model = model.to(device)
lam = 5


optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=5e-5)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

losses = []
model.train()
optimizer.zero_grad()
epochs = 10

for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for orig_acts in act_dl:
        orig_acts = orig_acts.to(device)
        feats, recon_acts = model(orig_acts)
        l2 = torch.sum(torch.square(orig_acts - recon_acts), dim=-1)
        l1 = torch.zeros(feats.shape[0]).to(device)
        for i in range(feats.shape[1]):
            l1 += (torch.abs(feats[:, i]) * torch.linalg.norm(model.dec.weight[:, i]))
        l1 = lam * l1
        loss = torch.mean(l2 + l1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        losses.append(loss.item())
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")
    # checkpoint(model, epoch, optimizer, loss.item(), run_name)

torch.save(model, f"{run_name}.pth")

losses = np.array(losses)
np.save(f'loss_{run_name}', losses)

plt.figure()
plt.plot(losses)
plt.title(f'loss: sae layer 2 conv1')
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig(f'loss_sae2conv1.pdf')