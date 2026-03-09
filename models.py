import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Conv2DBlock, GLUConvBlock, AttentionBlock, CapsuleLayer, SubPixelConv2d, NMFLayer
import torchinfo
from layers import *

class Generator(nn.Module):
    def __init__(self, input_channels=1, base_channels=8, num_downsample=4, rates=[5,7,9,11], frame_size=51):
        super().__init__()
        
        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 512, kernel_size=(128,3), stride=1),
                nn.InstanceNorm2d(512),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(512,256, kernel_size=(1,5), stride=1),
                nn.InstanceNorm2d(256),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=(1,5), stride=1),
                nn.InstanceNorm2d(128),
                nn.SiLU()
            )
        ])

        # self.exp_dim = 

        # self.bottleneck = DCNMFEncoder(128,frame_size,180)
        # self.vit = ConvAttention(128,180,90)

        # self.con_dim =

        #self.self_attn_ = nn.MultiheadAttention(512, num_heads=128)

        self.conv_t1d = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.InstanceNorm2d(128),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=5, dilation=5),
                nn.InstanceNorm2d(256),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=7, dilation=7),
                nn.InstanceNorm2d(512),
                nn.SiLU()
            )
        ])

        # self.conformer_t1d = ConformerBlockL(128, num_heads=16)

        #self.pix_shuf = SubPixelConv2d(base_channels,base_channels,2)

        self.postnet = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0),
                                     nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
                                     nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0),
                                     nn.BatchNorm2d(1),
                                     )

        self.hyp = nn.Softplus()#HypSnake(shape=(128,51))
        self.prob = nn.Softmax(dim=1)

    def reconstruct(self,H,W):
        pad = (W.shape[2]-1,W.shape[3]-1)
        rec = F.conv2d(H,W.flip((2,3)),padding=pad)
        return rec
        
    def forward(self, x, target_speaker_embedding=None):
        # Store skip connections
        res = x
        x = x
        # conv2 = self.conv2(x)
        # x = conv2.permute(0, 3, 1, 2)
        # flat = self.flat(x)
        # con = self.conformer(flat)

        # Store skip connections from encoder path
        skip_connections = []
        # x = con.permute(0, 2, 1)
        
        # Encoder path with skip connections
        for i,conv in enumerate(self.conv1):
            x = conv(x)
            # if i == 2:
            #     x = x #+ 0.1*res
            skip_connections.append(x)
        
        # Middle part
        # x, _ = self.self_attn(x, x, x)
        # W, H = self.bottleneck(x.unsqueeze(1))
        # rec = self.reconstruct(torch.sum(H, dim=0, keepdim=True), W).squeeze()
        # rec = rec.permute(0, 2, 1)
        # x, _ = self.self_attn_spectral(rec, rec, rec)
        # x = x.permute(0, 2, 1)

        # Decoder path with skip connections
        # for i, conv in enumerate(self.conv_t1d):
        #     x = conv(x)
        #     # Add skip connection if available
        #     if i < len(skip_connections):
        #         #print(skip_connections[-(i+1)].shape,x.shape)
        #         x = x + skip_connections[-(i+1)]  # Add from end of skip_connections list

        # # x = self.conformer_t1d(x.permute(0, 2, 1)) + con
        # x = self.postnet(x) #+ 0.1*flat.permute(0, 2, 1)
        
        # out = self.hyp(x) + 0.5*res  + torch.tensor(1e-8)
        # W = self.prob(x)
        return x,x
    

gen = Generator().cuda()
torchinfo.summary(gen, input_size=(32, 1, 128, 51))

#capsule network based patch-gan discriminator
class Discriminator(nn.Module):
    def __init__(self, rates=[3,5,7,9,11]):
        super(Discriminator, self).__init__()
        self.first = nn.Sequential(
            *[ResidualDilatedConvBlock(128,256,3,rates[i],51) if i==0 else ResidualDilatedConvBlock(256,256,7,rates[i],51) for i in range(len(rates))]
        )
        self.depth_point = nn.Sequential(
            nn.Conv1d(256,512,kernel_size=9,stride=1,groups=256),
            nn.InstanceNorm1d(512),
            nn.Conv1d(512,512,kernel_size=1,stride=1,padding=0),
            nn.Tanh()
        )
        self.capsnet = CapsuleNetwork()
        #self.final = nn.Conv1d(1,1,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x = self.first(x)
        x = self.depth_point(x)
        x = self.capsnet(x.unsqueeze(1))
        return x
    
# disc = Discriminator().cuda()
# torchinfo.summary(disc, input_size=(2, 128, 51))


## vocoder architecture

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class iSTFT_generator(nn.Module):
    def __init__(self, nfft=1920, hop=480, rates=[3,5,7,9,11], win='hann',n_layers=2):
        super(iSTFT_generator,self).__init__()
        self.nfft = nfft
        self.hop = hop
        #self.win = get_window(win,self.nfft)

        self.dil_sum = nn.Sequential(
            *[ResidualDilatedConvBlock(128,256,3,rates[i],51) if i==0 else ResidualDilatedConvBlock(256,256,7,rates[i],51) for i in range(len(rates))]
        )
        
        self.con = nn.Sequential(*[ConformerBlock(256,8,ff_expansion_factor=2) for _ in range(n_layers)])
        self.post = weight_norm(nn.Conv1d(256,self.nfft+2,15,1,padding=7))
        self.post.apply(init_weights)
        self.act = HypSnake(shape=(256,51))
        
    def forward(self,x):
        x = self.dil_sum(x)
        x = x.permute(0,2,1)
        con = self.con(x)
        con = self.act(con.permute(0,2,1))
        last = self.post(con)
        
        # Add numerical stability to prevent NaN
        last = torch.clamp(last, min=-20, max=20)  # Prevent extreme values before exp
        
        # Compute magnitude with stable exp
        spec = torch.exp(last[:,:self.nfft//2+1,:])
        spec = torch.clamp(spec, min=1e-7)  # Prevent zero magnitudes
        
        # Compute phase with bounded values
        phase = torch.tanh(last[:,self.nfft//2+1:,:])  # Use tanh instead of sin for bounded output
        
        return spec, phase
