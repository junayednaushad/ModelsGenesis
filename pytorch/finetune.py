import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


model = unet3d.UNet3D()

#Load pre-trained weights
weight_dir = './Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)

model.to(device)
print(device)