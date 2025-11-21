import argparse
from pathlib import Path
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
assert timm.__version__ == "0.4.5"  # version check
import timm.optim.optim_factory as optim_factory
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda import amp
import torch.distributed as dist

import models_mae
from util.load_data import *
import util.load_data
import util.loss
import util.trans
from util.test_save import record


import fusion
import connect_net


def get_args_parser():
    parser = argparse.ArgumentParser('MAE Fusion', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_decoder_4_640', type=str, metavar='MODEL',
                        help='Model Architecture')

    parser.add_argument('--input_size', default=640, type=int,
                        help='images input size LOCK!!!')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', default=1e-4,type=float,  metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='path/to/your/data', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='path/to/your/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./TMP',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0,1,2,3,4,5,6,7',
                        help='device to use for training')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='output_decoder_4_640/checkpoint-50.pth',
                        help='Pre-training ckpt for MAE(4 layers decoder)')
    parser.add_argument('--fusion_weight', default='path/to/fusion/layer/wegt',
                        help='ckpt for fusion layer (w/o)')
    parser.add_argument('--all_weight', default='path/to/fusion/all/wegt',
                        help='ckpt for all layer (w/o)')
    

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--training_mode',default="train_decoder_CFM_MFM",type=str,choices=['train_CFM_mean','train_CFM_fusion','train_MFM_mean_CFM_lock','train_MFM_fusion_CFM_lock',
                                                              'train_MFM_mean_CFM_open','train_MFM_fusion_CFM_open','train_decoder_only','train_decoder_MFM','train_decoder_CFM_MFM'],
                        help='training_mode')

    return parser

def trans_img(img, std, mean):
    img = (img * std + mean)*255
    return img

def train(args):

    #writer = SummaryWriter(args.log_dir)  
    
    #GPU
    device_ids = [int(device) for device in ((args.device).split(':')[1]).split(',')]
    devices = [torch.device(f'cuda:{device_id}') for device_id in device_ids]
    device  = devices[0]
    
    #load_model
    mae = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    
    fusion_layer = fusion.cross_fusion_FFN(embed_dim=1024, mode=args.training_mode) #same as mae
    """
    Note:
    If you want to load the fusion layer weights trained at a certain stage, please use the following code to load them.
    fusion_layer_wegt = torch.load(args.fusion_weight, map_location="cpu")
    fusion_layer.load_state_dict(fusion_layer_wegt['model'], strict=True)
    """
        
    #load_weight
    """
    "Resume" refers to the overall weights of both the encoder and decoder after retraining the decoder.
    """
    checkpoint_mae = torch.load(args.resume, map_location="cpu")
    mae.load_state_dict(checkpoint_mae['model'], strict=True)
    
    connect = connect_net.ConnectModel_MAE_Fuion(mae, fusion_layer, mode=args.training_mode)
    connect = connect.to(device)
    connect = nn.DataParallel(connect, device_ids=device_ids)
    """
    Note:
    If you want to load the all weights trained at a certain stage, please use the following code to load them.
    all_wegt = torch.load(args.all_weight, map_location="cpu")
    connect.load_state_dict(all_wegt['model'], strict=True)
    """

    #transform
    data_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #load_data
    matching_dataset = MatchingImageDataset_GRAY(root_dir=args.data_path, transform=data_transform)
    train_loader = DataLoader(matching_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    connect = torch.nn.DataParallel(connect)

    scaler = amp.GradScaler(enabled=args.device)
    param_groups = optim_factory.add_weight_decay(connect, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    Loss = util.loss.Fusionloss(1,2).to(device)
    Loss = nn.DataParallel(Loss,device_ids=device_ids)
    
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_std = torch.tensor([0.229, 0.224, 0.225])
    mean = image_mean.view(1, 3, 1, 1).to(device)
    std = image_std.view(1, 3, 1, 1).to(device)


    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch %d.....' % epoch)
        # load training database
        loss_out_fusion = 0
        loss_out_grad = 0
        loss_out_laplacian = 0
        count    = 0
        for imgs_vi, imgs_ir in tqdm(train_loader):
        

            connect.train()
            optimizer.zero_grad()
         
            imgs_vi = imgs_vi.to(device)
            imgs_ir = imgs_ir.to(device)

            result = connect(imgs_vi, imgs_ir)
            

            if args.training_mode == 'train_CFM_mean' or args.training_mode == 'train_MFM_mean_CFM_lock' or args.training_mode == 'train_MFM_mean_CFM_open':
                vi_latent = connect.module.model1.forward_encoder(imgs_vi)
                ir_latent = connect.module.model1.forward_encoder(imgs_ir)
                loss_total  = torch.mean((result - (vi_latent+ir_latent)/2)**2)

            else:
                result = connect.module.model1.unpatchify(result)
                recover = trans_img(result, std, mean)  #output_imgs 
                imgs_vi = trans_img(imgs_vi, std, mean) #vi_imgs
                imgs_ir = trans_img(imgs_ir, std, mean) #ir_imgs
                loss_total, loss_grad, loss_laplacian = Loss(recover ,imgs_vi ,imgs_ir)
            
            """
            The stage-one mean alignment loss function are: 
            loss_total  = torch.mean((result - (imgs_vi_latent+imgs_ir_latent)/2)**2)
            The stage-two fusion loss function are:
            recover = trans_img(result, std, mean)  #output_imgs 
            imgs_vi = trans_img(imgs_vi, std, mean) #vi_imgs
            imgs_ir = trans_img(imgs_ir, std, mean) #ir_imgs
            loss_total, loss_grad, loss_laplacian = Loss(recover ,imgs_vi ,imgs_ir)
            """
            
            
                
            loss = torch.mean(loss_total)
                
            loss_out_fusion += torch.mean(loss_total).item()
            # loss_out_grad += torch.mean(loss_grad).item()
            # loss_out_laplacian += torch.mean(loss_laplacian).item()

            scaler.scale(loss).backward() # type: ignore
            optimizer.step()

            count += 1
            
        print('total: ',loss_out_fusion/count)#,'laplacian: ',loss_out_laplacian/count,'grad: ',loss_out_grad/count)
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            """
            Note：
            If you only want to download the Fusion_layer weights, use：  model = fusion_layer.module
            If you want to retain the overall weights, use： model = connect.module
            """    
            model = connect.module
            torch.save({'model':model.state_dict()}, args.output_dir+'/checkpoint'+str(epoch)+'.pth')  




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        train(args)
