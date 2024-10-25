import torch
import torch.nn as nn

# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
# imagenet_std = torch.tensor([0.229, 0.224, 0.225])
# imagenet_mean_expanded = imagenet_mean.view(1, 3, 1, 1)
# imagenet_std_expanded = imagenet_std.view(1, 3, 1, 1)

# def RGB2YCrCb(img):
#     img = (img * imagenet_std_expanded + imagenet_mean_expanded)*255
#     Y = img[:,0,:,:]*0.257 + img[:,1,:,:]*0.504 + img[:,2,:,:]*0.098 + 16
#     return Y.unsqueeze(1)

# def RGB2GRAY(img):
#     img = (img * imagenet_std_expanded + imagenet_mean_expanded)*255
#     G = img[:,0,:,:]*0.2989 + img[:,1,:,:]*0.5870 + img[:,2,:,:]*0.1140
#     return G.unsqueeze(1)

class trans_img(nn.Module):
    def __init__(self):
        super(trans_img, self).__init__()
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        self.imagenet_mean_expanded = imagenet_mean.view(1, 3, 1, 1)
        self.imagenet_std_expanded = imagenet_std.view(1, 3, 1, 1)

    def forward(self,img):
        img = (img * self.imagenet_std_expanded + self.imagenet_mean_expanded)*255
        return img
    