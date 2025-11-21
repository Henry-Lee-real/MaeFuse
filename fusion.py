import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by num_heads"

        self.values = nn.Linear(embed_dim, embed_dim)
        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Split input into self.num_heads different parts
        batch_size, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        attention = torch.nn.functional.softmax(energy, dim=-1)

        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embed_dim)

        return out
    


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        
        self.downsample1 = self.downsample_block(in_channels, in_channels)
        self.downsample2 = self.downsample_block(in_channels, in_channels)
        
        self.upsample2 = self.upsample_block(in_channels, out_channels)
        self.upsample1 = self.upsample_block(out_channels, out_channels)
        
    def downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)
        
        x_up2 = self.upsample2(x2)
        x_up1 = self.upsample1(x_up2 + x1)
        
        return x_up1


class Feature_Net(nn.Module):
    def __init__(self, dim):
        super(Feature_Net, self).__init__()
        
        # 第一层卷积层：输入通道数1024，输出通道数2048，卷积核大小3x3，填充1
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1   = nn.BatchNorm2d(num_features=dim*2)
        
        # 第二层卷积层：输入通道数2048，输出通道数2048，卷积核大小3x3，填充1
        self.conv2 = nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2   = nn.BatchNorm2d(num_features=dim*2)
        
        # 第三层卷积层：输入通道数2048，输出通道数1024，卷积核大小3x3，填充1
        self.conv3 = nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3   = nn.BatchNorm2d(num_features=dim*2)
        
        # 第四层卷积层：输入通道数1024，输出通道数1024，卷积核大小3x3，填充1
        self.conv4 = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        #self.bn4   = nn.BatchNorm2d(num_features=dim)
        
    def forward(self, x):
        # 前向传播过程
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))
        return x
    

class cross_fusion(nn.Module):  #tradition cross*2
    def __init__(self,embed_dim):
        super(cross_fusion, self).__init__()
        self.cross_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.merge_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1600, embed_dim), requires_grad=False) 
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
        self.fc2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(self.embed_dim)
        self.initialize_weights()
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 40, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, vi_latent, ir_latent):
        vi_latent += self.pos_embed
        ir_latent += self.pos_embed
        vi_patten,_  = self.cross_model(ir_latent,vi_latent,vi_latent)  #q,k,v
        ir_patten,_  = self.cross_model(vi_latent,ir_latent,ir_latent)
        patten = vi_patten + ir_patten
        vi_final,_   = self.merge_model(patten,vi_latent,vi_latent)
        ir_final,_   = self.merge_model(patten,ir_latent,ir_latent)
        fusion    = (vi_final + ir_final)
        ffn = self.fc2(self.gelu(self.fc1(fusion)))
        out = self.ln(ffn+fusion)
        
        return out
    
class Seg_module(nn.Module):
    def __init__(self, embed_dim, img_size):
        super(Seg_module, self).__init__()
        #self.FPN = FPN(embed_dim, embed_dim)

        self.embed_dim = embed_dim
        self.decoder_embed_dim = 512
        self.patch_size = 16
        self.patch_embed = PatchEmbed(img_size, 16, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(4)])

        self.decoder_norm = nn.LayerNorm(512)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, 16**2 * 3, bias=True)
    
    def decoder(self, x):
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


    def unpatchify(self,x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, self.embed_dim))#1024
        x = torch.einsum('nhwc->nchw', x)
        
        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        
        h = w = imgs.shape[2]
        x = torch.einsum('nchw->nhwc', imgs)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.embed_dim))#1024
        return x
    
    def trans_mask(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] # type: ignore
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, latent):
        # imgs       = self.unpatchify(latent)
        # feature    = self.FPN(imgs)
        # new_latent = self.patchify(feature)
        seg_latent = self.decoder(latent)
        #mask       = self.trans_mask(seg_latent)
        return seg_latent
    

class cross_fusion_mean(nn.Module):
    def __init__(self,embed_dim):
        super(cross_fusion_mean, self).__init__()
        self.compare_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1600, embed_dim), requires_grad=False) 
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
        self.fc2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.initialize_weights()
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 40, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, vi_latent, ir_latent):#, com_latent
        #num         = vi_latent.shape[1]
        vi_latent += self.pos_embed
        ir_latent += self.pos_embed
        mean_latant = (vi_latent+ir_latent)/2
        vi_mean,_ = self.compare_model(mean_latant,vi_latent,vi_latent)
        ir_mean,_ = self.compare_model(mean_latant,ir_latent,ir_latent)
        latent = self.fc2(self.gelu(self.fc1(vi_mean+ir_mean)))
        
        return latent
    
class cross_fusion_FFN(nn.Module):  #tradition cross*2
    def __init__(self,embed_dim, mode='eval'):
        super(cross_fusion_FFN, self).__init__()
        self.cross_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.merge_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1600, embed_dim), requires_grad=False) 
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
        self.fc2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(self.embed_dim)
        self.initialize_weights()
        self.mode = mode
        self._set_trainable_blocks(self.mode)
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 40, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def _set_trainable_blocks(self, mode):
        if mode == 'train_MFM_mean_CFM_lock' or mode == 'train_MFM_fusion_CFM_lock' or mode == 'train_decoder_MFM':
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, vi_latent, ir_latent):#, com_latent
        #num         = vi_latent.shape[1]
        vi_latent += self.pos_embed
        ir_latent += self.pos_embed
        vi_patten,_  = self.cross_model(ir_latent,vi_latent,vi_latent)  #q,k,v
        ir_patten,_  = self.cross_model(vi_latent,ir_latent,ir_latent)
        patten = vi_patten + ir_patten
        if self.mode == 'train_CFM_mean' or self.mode == 'train_CFM_fusion':
            return patten
        vi_final,_   = self.merge_model(patten,vi_latent,vi_latent)
        ir_final,_   = self.merge_model(patten,ir_latent,ir_latent)
        fusion    = (vi_final + ir_final)
        ffn = self.fc2(self.gelu(self.fc1(fusion)))
        out = self.ln(ffn+fusion)
        return out
    
    
class cross_fusion_FFN_re(nn.Module):  #tradition cross*2
    def __init__(self,embed_dim):
        super(cross_fusion_FFN_re, self).__init__()
        self.cross_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.merge_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1600, embed_dim), requires_grad=False) 
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
        self.fc2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.vi1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
        self.vi2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
        self.ir1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
        self.ir2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
        self.initialize_weights()
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 40, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, vi_latent, ir_latent):#, com_latent
        #num         = vi_latent.shape[1]
        vi_latent += self.pos_embed
        ir_latent += self.pos_embed
        vi_patten,_  = self.cross_model(ir_latent,vi_latent,vi_latent)  #q,k,v
        ir_patten,_  = self.cross_model(vi_latent,ir_latent,ir_latent)
        patten = vi_patten + ir_patten
        vi_final,_   = self.merge_model(patten,vi_latent,vi_latent)
        ir_final,_   = self.merge_model(patten,ir_latent,ir_latent)
        fusion    = (vi_final + ir_final)
        out = self.fc2(self.gelu(self.fc1(fusion)))
        
        return out
