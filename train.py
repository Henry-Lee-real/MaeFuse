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
                        help='Name of model to train')#EFD:mae_encoder #tradition:mae_decoder_4_640 

    parser.add_argument('--input_size', default=640, type=int,
                        help='images input size')

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
    parser.add_argument('--data_path', default='/data_b/lijy/TRAIN_NEW', type=str,
                        help='dataset path')#/data_b/lijy/mae_seg/MSRS-main/train#/data_b/lijy/TRAIN_NEW

    parser.add_argument('--output_dir', default='./cross_ffn_1_2_TRAIN_NEW_open_32_2',#m3fd_20_finetune_lapas_1
                        help='path where to save, empty for no saving')#cross_ffn_1_2_TRAIN_NEW_open_32
    parser.add_argument('--log_dir', default='./TMP',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0,1,2,3,4,5,6,7',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='output_decoder_4_640/checkpoint-50.pth',#output_decoder_4_640/checkpoint-50.pth
                        help='resume from checkpoint')
    parser.add_argument('--seg', default='seg_checkpoint99.pth',
                        help='resume from checkpoint')
    parser.add_argument('--fusion_weight', default='mean_att_fc_fusion_align/checkpoint50.pth',
                        help='resume from checkpoint')#fusion_layer/checkpoint9.pth
    parser.add_argument('--core_weight', default='mae_640_core_cross/checkpoint39.pth',
                        help='resume from checkpoint')
    parser.add_argument('--core', default=True,
                        help='whether use core')
    

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
    
    # encoder = models_encoder.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    # decoder = Decoder.Decoder(embed_dim=1024) #same as mae

    # #load_weight

    # connectED = connect_net.ED(encoder,decoder)
    
    # checkpoint_ED = torch.load("ED_2*FC_Unet/checkpoint20.pth", map_location="cpu")
    # connectED.load_state_dict(checkpoint_ED['model'], strict=True)
    
    
    fusion_layer = fusion.cross_fusion_FFN(embed_dim=1024) #same as mae
    checkpoint_connect = torch.load("cross_ffn_align_new/checkpoint20.pth", map_location="cpu")
    fusion_layer.load_state_dict(checkpoint_connect['model'], strict=True)
    
    

    #load_weight
    checkpoint_mae = torch.load(args.resume, map_location="cpu")
    mae.load_state_dict(checkpoint_mae['model'], strict=True)
    
    # pretrained_dict = torch.load(args.fusion_weight, map_location='cpu')
    # fusion_layer_dict = fusion_layer.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict['model'].items() if k in ['pos_embed','cross_model_vi','cross_model_ir','merge_model']}
    # fusion_layer_dict.update(pretrained_dict)
    # fusion_layer.load_state_dict(fusion_layer_dict)
    

    # checkpoint_fusion = torch.load(args.fusion_weight, map_location='cpu')
    # fusion_layer.load_state_dict(checkpoint_fusion['model'], strict=True)
    # if args.core:
    #     checkpoint_core = torch.load(args.core_weight, map_location='cpu')
    #     fusion_layer.cross_model.load_state_dict(checkpoint_core['model']['vi_model'], strict=True) # type: ignore
    # checkpoint_seg = torch.load(args.seg, map_location="cpu")
    # seg_module.load_state_dict(checkpoint_seg['model'], strict=True)
    
    connect = connect_net.ConnectModel_MAE_Fuion(mae, fusion_layer)
    
    connect = connect.to(device)
    connect = nn.DataParallel(connect, device_ids=device_ids)

    #transform
    data_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #load_data
    matching_dataset = MatchingImageDataset_GRAY(root_dir=args.data_path, transform=data_transform)
    train_loader = DataLoader(matching_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    #image_names = util.load_data.get_address(args.data_path)

    #connect = torch.nn.DataParallel(connect)

    scaler = amp.GradScaler(enabled=args.device)
    param_groups = optim_factory.add_weight_decay(connect, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    Loss = util.loss.Fusionloss(1,2).to(device)
    Loss = nn.DataParallel(Loss,device_ids=device_ids)
    
    # trans_img = util.trans.trans_img().to(device)
    # trans_img = nn.DataParallel(trans_img,device_ids=device_ids)
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_std = torch.tensor([0.229, 0.224, 0.225])
    mean = image_mean.view(1, 3, 1, 1).to(device)
    std = image_std.view(1, 3, 1, 1).to(device)


    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch %d.....' % epoch)
        # load training database
        #image_random ,batches = util.load_data.load_dataset(image_names ,args.batch_size)
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
            result = connect.module.model1.unpatchify(result)
            recover = trans_img(result, std, mean)
            imgs_vi = trans_img(imgs_vi, std, mean)
            imgs_ir = trans_img(imgs_ir, std, mean)
            loss_total, loss_grad, loss_laplacian = Loss(recover ,imgs_vi ,imgs_ir)
            # loss_total  = torch.mean((result - ta)**2)
                
            loss = torch.mean(loss_total)
                
            loss_out_fusion += torch.mean(loss_total).item()
            loss_out_grad += torch.mean(loss_grad).item()
            loss_out_laplacian += torch.mean(loss_laplacian).item()

            
            scaler.scale(loss).backward() # type: ignore
            optimizer.step()

            count += 1
            
        print('total: ',loss_out_fusion/count,'laplacian: ',loss_out_laplacian/count,
              'grad: ',loss_out_grad/count)
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            record(task_name=args.output_dir.split('/')[1],
                   epoch=epoch,
                   val_file="./val",
                   model = connect,
                   size = args.input_size)
            if epoch + 1 == args.epochs or epoch == 0 or epoch%5 == 0:
                model = connect.module
                torch.save({'model':model.state_dict()}, args.output_dir+'/checkpoint'+str(epoch)+'.pth')  # 保存权重到.pth文件




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        train(args)
