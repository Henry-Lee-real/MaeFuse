import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def framework_loss(pred ,vi ,ir):
    l1_loss = torch.nn.L1Loss()
    MSELoss = torch.nn.MSELoss()  
    return l1_loss(pred,vi) + MSELoss(pred,ir)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()      

    def forward(self,generate_img,image_vis,image_ir):
        image_y=image_vis
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        new_kernelx = torch.zeros((1, 3, 3, 3))
        new_kernely = torch.zeros((1, 3, 3, 3))
        for i in range(3):
            new_kernelx[0, i, :, :] = kernelx[0, 0, :, :]
        for i in range(3):
            new_kernely[0, i, :, :] = kernely[0, 0, :, :]
        self.weightx = nn.Parameter(data=new_kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=new_kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)