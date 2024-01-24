
import torch
import torch.nn as nn



class ConnectModel_decoder4(nn.Module):
    def __init__(self, model1, model2):
        super(ConnectModel_decoder4, self).__init__()
        self.model1 = model1
        self.model2 = model2
        

    def forward(self, vi, ir):
        vi_latent = self.model1.forward_encoder(vi)
        ir_latent = self.model1.forward_encoder(ir)
        fusion    = (vi_latent+ir_latent)/2#self.model2(vi_latent,ir_latent)
        out       = self.model1.forward_decoder(fusion)
        return out 
    
# class ConnectModel_seg(nn.Module):
#     def __init__(self, model1):
#         super(ConnectModel_seg, self).__init__()
#         self.model1 = model1
#         self.model2 = model2
        

#     def forward(self, vi, ir):
#         vi_latent = self.model1.forward_encoder(vi)
#         ir_latent = self.model1.forward_encoder(ir)
#         fusion    = self.model2(vi_latent,ir_latent)
#         out       = self.model1.forward_decoder(fusion)
#         return out 
