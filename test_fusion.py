import numpy as np
import cv2
import models_mae
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

import fusion
from connect_net import ConnectModel_MAE_Fuion
from util.trans import trans_img
from util.load_data import *
from tqdm import tqdm

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def prepare_model(arch='mae_decoder_2'):
    # build model
    model = models_mae.__dict__[arch](norm_pix_loss=False)
    return model


def prepare_image(img):
    x = torch.tensor(img)
    #print(x)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    return x.float().to(device)


def show_image(image, title=''):
    # image is [H, W, 3]
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def normalize_image(img):
    img = img / 255.
    assert img.shape == (640, 640, 3)
    img = img - imagenet_mean
    img = img / imagenet_std
    return img


def load_model():
    model = prepare_model(arch="mae_decoder_4_640")
    fusion_layer = fusion.cross_fusion(embed_dim=1024)
    checkpoint = torch.load("final_new_60.pth", map_location='cpu')
    connect = ConnectModel_MAE_Fuion(model,fusion_layer)
    connect.load_state_dict(checkpoint['model'], strict=True)
    connect = connect.to(device)
    connect.eval()
    return connect



if __name__ == '__main__':

    connect = load_model()
    
    address = "TNO_dataset"
    
    names = get_address(address)
    type = get_image_type(names[0],address+"/vi/")



    folder_path = "TNO_55_final_2"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    transform_to_pil = transforms.ToPILImage()

    for name in tqdm(names):
        img_vi = cv2.imread(address+'/vi/'+ name + type)
        img_ir = cv2.imread(address+'/ir/'+ name + type)
        img_vi_ycrcb = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
        img_vi = cv2.cvtColor(img_vi_ycrcb[:,:,0], cv2.COLOR_GRAY2BGR)
        img_vi = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)
        h,w,_ = img_vi.shape
        target_size = (w, h)
                
                
        img_vi = cv2.resize(img_vi,(640,640),cv2.INTER_CUBIC)
        img_ir = cv2.resize(img_ir,(640,640),cv2.INTER_CUBIC)

        img_vi = normalize_image(img_vi)
        img_ir = normalize_image(img_ir)

        
        result = connect(prepare_image(img_vi), prepare_image(img_ir))

        pred = connect.model1.unpatchify(result)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        assert pred[0].shape[2] == 3
        pred = torch.clip((pred[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int()

        path = folder_path+"/"+name+type
        #image_pil.save(file_path)

        pred = pred.numpy()
        pred = cv2.convertScaleAbs(pred)
        y = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        img_vi_ycrcb = cv2.resize(img_vi_ycrcb,(640,640),cv2.INTER_CUBIC) # type: ignore
        img_vi_ycrcb[:,:,0] = y
        out = cv2.cvtColor(img_vi_ycrcb, cv2.COLOR_YCrCb2BGR)
        out = cv2.resize(out,target_size,interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path,out)