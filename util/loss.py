import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.losses


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.window_size  = 11
        self.window_sigma = 1.5
        self.k1           = 0.01
        self.k2           = 0.03

    def gaussian(self, x, y):
        window_size  = self.window_size
        window_sigma = torch.tensor(self.window_sigma)
        return torch.exp(-((x - window_size // 2) ** 2 + (y - window_size // 2) ** 2) / (2 * window_sigma ** 2))

    def create_window(self):
        window_size  = self.window_size
        window = torch.tensor([[
            [self.gaussian(i, j) for j in range(window_size)]
            for i in range(window_size)
        ]], dtype=torch.float32)

        return window / window.sum()
    
    def rgb_to_y(self,rgb_image):
        
        r = rgb_image[:, 0, :, :]
        g = rgb_image[:, 1, :, :]
        b = rgb_image[:, 2, :, :]     
        y_image = 0.299 * r + 0.587 * g + 0.114 * b  
        
        return y_image
    
    def forward(self, y_pred, y_target):

        y_pred   = self.rgb_to_y(y_pred)
        y_target = self.rgb_to_y(y_target)

        window_size  = self.window_size
        k1           = self.k1 
        k2           = self.k2 

        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2

        mu_x = F.conv2d(y_pred, self.create_window(), padding=window_size // 2,stride=1)
        mu_y = F.conv2d(y_target, self.create_window(), padding=window_size // 2,stride=1)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x_sq = F.conv2d(y_pred * y_pred, self.create_window(), padding=window_size // 2, stride=1) - mu_x_sq
        sigma_y_sq = F.conv2d(y_target * y_target, self.create_window(), padding=window_size // 2, stride=1) - mu_y_sq
        sigma_xy = F.conv2d(y_pred * y_target, self.create_window(), padding=window_size // 2,stride=1) - mu_x_mu_y

        ssim_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = ssim_n / ssim_d
        return ssim_map.mean()

def rgb_to_gray(images):
    weight = torch.tensor([[[[0.2989]]], 
                           [[[0.5870]]], 
                           [[[0.1140]]]], dtype=torch.float32)

    # Move the weight tensor to the same device as the images
    weight = weight.to(images.device)

    # Applying the depthwise convolution
    gray_images = F.conv2d(images, weight=weight, groups=3)

    # Since the output will have 3 separate channels, sum them up across the channel dimension
    gray_images = gray_images.sum(dim=1, keepdim=True)

    return gray_images


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
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()
        kernel = [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        # Apply convolution with the Laplacian kernel
        laplacian = F.conv2d(x, self.weight, padding=1)
        return laplacian


class Fusionloss(nn.Module):
    def __init__(self,alpha,beta):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy() 
        self.laplacianconv = Laplacian()
        # self.Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        self.alpha = alpha  
        self.beta  = beta 

    def forward(self,generate_img,image_vis,image_ir):
        
        
        generate_img = rgb_to_gray(generate_img)
        image_vis = rgb_to_gray(image_vis)
        image_ir  = rgb_to_gray(image_ir)
        
        # x_in_max = torch.max(image_vis,image_ir)
        # loss_in  = F.l1_loss(x_in_max,generate_img)

        y_grad            = self.sobelconv(image_vis)
        ir_grad           = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint      = torch.max(y_grad,ir_grad)
        loss_grad         = F.l1_loss(x_grad_joint,generate_img_grad)
        
        y_laplacian            = self.laplacianconv(image_vis)
        ir_laplacian           = self.laplacianconv(image_ir)
        generate_img_laplacian = self.laplacianconv(generate_img)
        x_laplacian_joint      = torch.max(y_laplacian,ir_laplacian)
        loss_laplacian         = F.l1_loss(x_laplacian_joint,generate_img_laplacian)

    

        loss_total = self.alpha*loss_grad + self.beta*loss_laplacian
        return loss_total,  loss_grad, loss_laplacian

