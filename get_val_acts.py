import torch
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
from einops import rearrange
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# gets neuron activations from resnet18 for all images in imagenet train set, save acts
# f = h5py.File('/scratch/gpfs/js5013/data/imagenet_train.hdf5', 'r')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valdata = torchvision.datasets.ImageNet(root="/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization",
                                          split="val", transform=transform)
batch_size = 2048
valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size,
                                          shuffle=False, num_workers=8, pin_memory=True)

def get_act(layer=1):
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
    
    # sample a few neurons in the 128 28x28 feature maps (conv layer2) and in the 512 7x7 feature maps (conv layer4)
    # pick center position and all channels, and also try all positions in 1 channel (for both layers)

    # shape: batch_size, channels, height, width

    # center pos layer 2: 13,13
    # center pos layer 4: 3,3
    if layer == 1:
        hook = resnet.layer1[0].conv1.register_forward_hook(get_activation('1conv1'))
        conv_num = 1
        cen = (56-1)//2
        ch = 64
        n = 56
    elif layer == 2:
        hook = resnet.layer2[0].conv1.register_forward_hook(get_activation('2conv1'))
        conv_num = 1
        cen = 13
        ch = 128
        n = 28
    elif layer == 4:
        hook = resnet.layer4[1].conv2.register_forward_hook(get_activation('4conv2'))
        conv_num = 2
        cen = 3
        ch = 512
        n = 7

    resnet.eval()

    output_file = h5py.File(f"./data/act_{layer}conv{conv_num}_val.hdf5", "w")
    output_file.create_dataset('cen', (len(valdata), ch), dtype="f")
    output_file.create_dataset('all', (len(valdata), (n*n)), dtype="f")

    # pass images through resnet to get activations
    for i, (inputs, _) in enumerate(tqdm(valloader)):
        inputs = inputs.to(device)

        with torch.no_grad():
            output = resnet(inputs)
            del inputs
            del output
            
            # collect the activations
            act = activation[f'{layer}conv{conv_num}']

            act_cen = act[:, :, cen, cen]
            act_all = rearrange(act, 'b c h w -> b c (h w)')
            act_all = act_all[:, 0, :]
    
            output_file["cen"][i * batch_size : (i + 1) * batch_size] = act_cen
            output_file["all"][i * batch_size : (i + 1) * batch_size] = act_all

    # detach the hooks
    hook.remove()
    output_file.close()

get_act(layer=1)
get_act(layer=2)
get_act(layer=4)