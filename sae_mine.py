import torch.nn as nn
import numpy as np
import torch
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mine
import argparse
from einops import rearrange

parser = argparse.ArgumentParser(description="from SAE features, find disentangled representations")

parser.add_argument('--path', type=str, default='/scratch/gpfs/ls1546/sae-mine/', help='path to script dir')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--image_size', type=int, default=224, help='image size')
parser.add_argument('--num_feats', type=int, default=700, help='num features in SAE')
parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
parser.add_argument('--run_name', type=str, default='', help='run name')
parser.add_argument('--type', type=str, default='cen', help='cen or all')
parser.add_argument('--layer', type=int, default=2, help='2 or 4')
args = parser.parse_args()

# for cen neuron, all channels, act2 has 128, act4 has 512, 700 feats
# for all neuron in 1 channel, act2 has 784, act4 has 49, 1000 feats

# params
path = args.path
datadir = f'{path}data/'
figdir = f'{path}figures/'
batch_size = args.batch_size
image_size = args.image_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_feats = args.num_feats
epochs = args.epochs
x_dim=image_size*image_size*3
y_dim=1
run_name = args.run_name
type = args.type
layer = args.layer

class ImageNetwork(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.fc1x = nn.Linear(x_dim, 1, bias=False)
        self.fc1y = nn.Linear(y_dim, 1, bias=True)
        self.fc2 = nn.Linear(2, 100, bias=True)
        self.fc3 = nn.Linear(100, 1, bias=True)

    def forward(self, x, y):
        x = F.relu(self.fc1x(x))
        y = F.relu(self.fc1y(y))
        h = torch.cat((x, y), dim=1)
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h
    
# split the responses and images into a train, val, and test set
def data_split(images_flat, act):
    train_samples = int(.8*len(images_flat))
    val_samples = int(.1*len(images_flat))

    images_flat_train = images_flat[:train_samples]
    act_train = act[:train_samples]

    images_flat_val = images_flat[train_samples:train_samples+val_samples]
    act_val = act[train_samples:train_samples+val_samples]

    images_flat_test = images_flat[train_samples+val_samples:]
    act_test = act[train_samples+val_samples:]

    img_train = torch.tensor(images_flat_train, dtype=torch.float32)
    act_train = torch.tensor(act_train, dtype=torch.float32)
    img_val = torch.tensor(images_flat_val, dtype=torch.float32)
    act_val = torch.tensor(act_val, dtype=torch.float32)
    img_test = torch.tensor(images_flat_test, dtype=torch.float32)
    act_test = torch.tensor(act_test, dtype=torch.float32)
    return img_train, act_train.unsqueeze(1), img_val, act_val.unsqueeze(1), img_test, act_test.unsqueeze(1)   
 
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

# load data
images_flat = np.load(f'{datadir}images_flat_mine_all.npy')
images_flat = rearrange(images_flat, 'n b c -> (n b) c')
if type == 'cen':
    if layer == 2:
        feat2cen = np.load(f'{datadir}feats_act2cen_all.npy')
        feat2cen = rearrange(feat2cen, 'n b c -> (n b) c')
    elif layer == 4:
        feat4cen = np.load(f'{datadir}feats_act4cen_all.npy')
        feat4cen = rearrange(feat4cen, 'n b c -> (n b) c')
    print('loaded cen')
else:
    if layer == 2:
        feat2all = np.load(f'{datadir}feats_act2all_all.npy')
        feat2all = rearrange(feat2all, 'n b c -> (n b) c')
    elif layer == 4:
        feat4all = np.load(f'{datadir}feats_act4all_all.npy')
        feat4all = rearrange(feat4all, 'n b c -> (n b) c')
    print('loaded all')

def train_mine(run_name, lam, img_train, act_train, epochs, batch_size, img_val, act_val):
    model = mine.Mine(
        T=ImageNetwork(x_dim, y_dim),
        loss="mine",  # mine_biased, fdiv
        device=device).to(device)

    mi, loss_list, loss_type = model.optimize(img_train, act_train, epochs, batch_size, 
                                            lam, run_name, img_val, act_val)

    torch.save(model.T, f"{datadir}{run_name}_mine.pt")
    np.save(f"{datadir}{run_name}_mi.npy", mi.detach().cpu().numpy())
    np.save(f"{datadir}{run_name}_loss.npy", loss_list)
    np.save(f"{datadir}{run_name}_loss_type.npy", loss_type)

    plt.figure()
    plt.plot(loss_list)
    plt.title(f"loss: {run_name}, {epochs} epochs")
    plt.ylabel("loss")
    plt.xlabel("batches")
    plt.savefig(f"{figdir}{run_name}_loss.pdf")
    plt.close()

    plt.figure()
    plt.plot(loss_type)
    plt.title(f"loss type: {run_name}, {epochs} epochs")
    plt.ylabel("loss type (0=mine_biased, 1=mine)")
    plt.xlabel("batches")
    plt.savefig(f"{figdir}{run_name}_loss_type.pdf")
    plt.close()

    Tweightsx = model.T.fc1x.weight.detach().cpu().numpy()
    Tweightsy = model.T.fc1y.weight.detach().cpu().numpy()
    np.save(f'{datadir}{run_name}_Tweightsx.npy', Tweightsx)
    np.save(f'{datadir}{run_name}_Tweightsy.npy', Tweightsy)


def mine_feats(num_feats, images_flat, feats, run_name, lam, epochs, batch_size):
    for f in tqdm(range(num_feats)):
        img_train, act_train, img_val, act_val, _, _ = data_split(images_flat, feats[:, f])
        run = f'{run_name}_f{f}'
        # print(run)
        train_mine(run, lam, img_train, act_train, epochs, batch_size, img_val, act_val)


lam = 0.005
if type == 'cen':
    # num_feats = 700
    if layer == 2:
        run = f'{run_name}_act2cen'
        mine_feats(num_feats, images_flat, feat2cen, run, lam, epochs, batch_size)
    elif layer == 4:
        run = f'{run_name}_act4cen'
        mine_feats(num_feats, images_flat, feat4cen, run, lam, epochs, batch_size)
else:
    # num_feats = 1000
    if layer == 2:
        run = f'{run_name}_act2all'
        mine_feats(num_feats, images_flat, feat2all, run, lam, epochs, batch_size)
    elif layer == 4:
        run = f'{run_name}_act4all'
        mine_feats(num_feats, images_flat, feat4all, run, lam, epochs, batch_size)