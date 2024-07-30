import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from einops import rearrange

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
    
act_4conv2 = torch.from_numpy(np.load('./data/act_4conv2.npy'))
act_2conv1 = torch.from_numpy(np.load('./data/act_2conv1.npy'))

class ConvActDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def train_sae(epochs, act_dl, run_name, device, model, lam, optimizer):
    losses = []
    model.train()
    optimizer.zero_grad()
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

    torch.save(model, f"./models/{run_name}.pt")

    losses = np.array(losses)
    np.save(f'./data/loss_{run_name}', losses)
    return losses

def plot_loss(losses, run_name, title):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig(f'./figures/loss_sae{run_name}.pdf')
    
# sample a few neurons in the 128 28x28 feature maps (conv layer2) and in the 512 7x7 feature maps (conv layer4)
# pick center position and all channels, and also try all positions in 1 channel (for both layers)
# get the activations from all images
batch_size = 128
# shape: num_batches, batch_size, channels, height, width

# trying all pos from first channel
act2_data = rearrange(act_2conv1, 'n b c h w -> (n b) c (h w)')
act2_data = act2_data[:, 0, :]

act4_data = rearrange(act_4conv2, 'n b c h w -> (n b) c (h w)')
act4_data = act4_data[:, 0, :]

act4_dataset = ConvActDataset(act4_data)
act2_dataset = ConvActDataset(act2_data)
act4_dl = torch.utils.data.DataLoader(act4_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
act2_dl = torch.utils.data.DataLoader(act2_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# train sae with acts from resnet hooks
act_dl = act4_dl
run_name = 'act4all'
d_act = next(iter(act_dl)).shape[-1]
d_feat = 1000
model = SAE(d_act, d_feat)
model = model.to(device)
lam = 5
epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=5e-5)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

losses = train_sae(epochs, act_dl, run_name, device, model, lam, optimizer)
title = 'loss: sae layer 4 conv2, center neuron(3,3)'
plot_loss(losses, run_name, title)

# train sae with acts from resnet hooks
act_dl = act2_dl
run_name = 'act2all'
d_act = next(iter(act_dl)).shape[-1]
d_feat = 1000
model = SAE(d_act, d_feat)
model = model.to(device)
lam = 5
epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=5e-5)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

losses = train_sae(epochs, act_dl, run_name, device, model, lam, optimizer)
title = 'loss: sae layer 2 conv1, center neuron(13,13)'
plot_loss(losses, run_name, title)