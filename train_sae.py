import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import argparse
import h5py

parser = argparse.ArgumentParser(description="train sae on resnet activations")

parser.add_argument('--num_feats', type=int, default=700, help='num features in SAE')
parser.add_argument('--type', type=str, default='cen', help='cen or all')
parser.add_argument('--layer', type=int, default=2, help='1,2, or 4')
parser.add_argument('--conv_num', type=int, default=1, help='1 or 2')
args = parser.parse_args()

num_feats = args.num_feats
type = args.type
layer = args.layer
conv_num = args.conv_num
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
    
def checkpoint(model, epoch, optimizer, loss, run_name):
    path = f"./models/ckpt{run_name}{epoch}.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)
    
def train_sae(epochs, act_dl, run_name, device, lam, num_feats):
    model = SAE(next(iter(act_dl)).shape[-1], num_feats)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=5e-5)
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
        checkpoint(model, epoch, optimizer, loss.item(), run_name)

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

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, ds_name):
        self.file_path = path
        self.ds_name = ds_name
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file[f"{self.ds_name}"])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[f"{self.ds_name}"]
        return self.dataset[index]

    def __len__(self):
        return self.dataset_len

def make_dl(layer, conv_num, type):
    h5path = f"./data/act_{layer}conv{conv_num}.hdf5"
    act_dataset = H5Dataset(h5path, type)
    act_dl = torch.utils.data.DataLoader(act_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return act_dl

# train sae with acts from resnet hooks
lam = 3.5e-6
epochs = 10
batch_size = 128
# type = ["cen"] or ["all"]

def train_and_plot(layer, conv_num, type, run_name, num_feats):
    act_dl = make_dl(layer, conv_num, type)
    losses = train_sae(epochs, act_dl, run_name, device, lam, num_feats)
    title = f'sae loss: {run_name}'
    plot_loss(losses, run_name, title)

train_and_plot(layer, conv_num, type, f'act{layer}{type}', num_feats)