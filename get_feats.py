import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="get feats from sae from resnet val acts")

parser.add_argument('--num_feats', type=int, default=700, help='num features in SAE')
parser.add_argument('--type', type=str, default='cen', help='cen or all')
parser.add_argument('--layer', type=int, default=2, help='1,2, or 4')
parser.add_argument('--conv_num', type=int, default=1, help='1 or 2')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
args = parser.parse_args()

num_feats = args.num_feats
type = args.type
layer = args.layer
conv_num = args.conv_num
batch_size = args.batch_size
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
    h5path = f"./data/act_{layer}conv{conv_num}_val.hdf5"
    act_dataset = H5Dataset(h5path, type)
    act_dl = torch.utils.data.DataLoader(act_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return act_dl

def get_feats(layer, conv_num, type, d_feat, run_name, model_name):
    act_dl = make_dl(layer, conv_num, type)
    d_act = next(iter(act_dl)).shape[-1]
    model = SAE(d_act, d_feat)
    model = torch.load(f'./models/{model_name}.pt')
    model = model.to(device)
    model.eval()
    allfeats = []
    allrecon_acts = []
    for orig_acts in tqdm(act_dl):
        orig_acts = orig_acts.to(device)
        feats, recon_acts = model(orig_acts)
        allfeats.append(feats.detach().cpu().numpy())
        allrecon_acts.append(recon_acts.detach().cpu().numpy())

    np.save(f'./data/feats_{run_name}', np.array(allfeats))
    np.save(f'./data/recon_acts_{run_name}', np.array(allrecon_acts))

run_name = f'act{layer}{type}'
get_feats(layer, conv_num, type, num_feats, run_name, run_name)