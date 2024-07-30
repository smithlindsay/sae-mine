import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

testdata = torchvision.datasets.ImageNet(root="/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization",
                                          split="val", transform=transform)
batch_size = 512
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                          shuffle=True, num_workers=8, pin_memory=True)

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
    
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

resnet.to(device)

# add hooks, run model with inputs to get activations

# a dict to store the activations
activation = {}
def get_activation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()
    return hook

hookl2 = resnet.layer2[0].conv1.register_forward_hook(get_activation('2conv1'))
hookl4 = resnet.layer4[1].conv2.register_forward_hook(get_activation('4conv2'))

inputs_list = []
# outputs_list = []
act_listl2 = []
act_listl4 = []
resnet.eval()

# pass images through resnet to get activations
for inputs, _ in tqdm(testloader):
    inputs = inputs.to(device)

    with torch.no_grad():
        output = resnet(inputs)
        
        # collect the activations
        act_listl2.append(activation['2conv1'])
        act_listl4.append(activation['4conv2'])
        inputs = torch.flatten(inputs, start_dim=1)
        inputs_list.append(inputs.detach().cpu().numpy())
        # outputs_list.append(output.detach().cpu().numpy())

    del inputs
    del output

# detach the hooks
hookl2.remove()
hookl4.remove()

act_listl2 = np.array(act_listl2[:-1])
act_listl4 = np.array(act_listl4[:-1])
inputs_list = np.array(inputs_list[:-1])

act_listl2.shape, act_listl4.shape, inputs_list.shape

np.save('images_flat_mine_all.npy', inputs_list)
np.save('act_4conv2_mine_all.npy', act_listl4)
np.save('act_2conv1_mine_all.npy', act_listl2)

act_2conv1 = torch.from_numpy(act_listl2)
act_4conv2 = torch.from_numpy(act_listl4)
images_flat = inputs_list

del act_listl2, act_listl4, inputs_list

class ConvActDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

cenl2 = 13
cenl4 = 3

act2_data = act_2conv1[:, :, :, cenl2, cenl2]
act2_data = rearrange(act2_data, 'n b c -> (n b) c')

act4_data = act_4conv2[:, :, :, cenl4, cenl4]
act4_data = rearrange(act4_data, 'n b c -> (n b) c')

act4_dataset = ConvActDataset(act4_data)
act2_dataset = ConvActDataset(act2_data)
act4_dl = torch.utils.data.DataLoader(act4_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
act2_dl = torch.utils.data.DataLoader(act2_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

def get_feats(act_dl, d_feat, run_name, model_name):
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

get_feats(act4_dl, 700, 'act4cen_all', 'act4cen')
get_feats(act2_dl, 700, 'act2cen_all', 'act2cen')

# trying all pos from first channel
act2_data = rearrange(act_2conv1, 'n b c h w -> (n b) c (h w)')
act2_data = act2_data[:, 0, :]

act4_data = rearrange(act_4conv2, 'n b c h w -> (n b) c (h w)')
act4_data = act4_data[:, 0, :]

act4_dataset = ConvActDataset(act4_data)
act2_dataset = ConvActDataset(act2_data)
act4_dl = torch.utils.data.DataLoader(act4_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
act2_dl = torch.utils.data.DataLoader(act2_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

get_feats(act4_dl, 1000, 'act4all_all', 'act4all')
get_feats(act2_dl, 1000, 'act2all_all', 'act2all')