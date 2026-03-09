import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class HypSnake(nn.Module):#learnable a
  def __init__(self,shape):
      super(HypSnake, self).__init__()
      self.a = nn.Parameter(torch.zeros(shape))
      self.first = True
  def forward(self, x):
    if self.first:
        self.first = False
        a = torch.zeros_like(x[0]).normal_(mean=0,std=50).abs()
        self.a = nn.Parameter(a)
    return torch.tanh(x + (torch.sin(self.a * x) ** 2) / self.a)

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class GLUConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels * 2)
        
    def forward(self, x):
        x = self.norm(self.conv(x))
        a, b = torch.chunk(x, 2, dim=1)
        return a * torch.sigmoid(b)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out + x_flat)
        return attn_out.transpose(1, 2).view(b, c, h, w)

class CapsuleLayer(nn.Module):
    def __init__(self, in_caps, in_dim, num_caps, cap_dim):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        self.attention = nn.MultiheadAttention(cap_dim, 4, batch_first=True)
        self.route_weights = nn.Parameter(torch.randn(in_caps, num_caps, in_dim, cap_dim))
        
    def forward(self, x):
        # x shape: [batch, in_caps, in_dim]
        batch = x.shape[0]
        
        # Transform input capsules
        x = x.unsqueeze(2) @ self.route_weights  # [batch, in_caps, num_caps, cap_dim]
        x = x.transpose(1, 2)  # [batch, num_caps, in_caps, cap_dim]
        
        # Attention-based routing
        x = x.reshape(batch * self.num_caps, -1, self.cap_dim)
        x, _ = self.attention(x, x, x)
        x = x.reshape(batch, self.num_caps, -1, self.cap_dim).mean(2)
        
        # Normalize capsule vectors
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        return x * F.softmax(x_norm, dim=1)

class SubPixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor ** 2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

# Non-negative Matrix Factorization inspired constraint
class NMFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_features, out_features))
        
    def forward(self, x):
        return F.relu(x @ F.relu(self.weight))
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=1, dilation=1)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size, padding=dilation, dilation=dilation)
    
    def forward(self, x):
        residual = x
        x = torch.nn.SiLU()(self.conv1(x)) 
        x = self.conv2(x)
        x = torch.nn.SiLU()(self.conv3(x))
        return x + residual
    
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation,frames):
        super(ResidualDilatedConvBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2))
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.SiLU()
        self.residual = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out += residual
        return (out)  


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv1d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, 
                                   padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, 
                                   padding=0, dilation=1, groups=out_channels, bias=bias)
        self.ins = nn.GroupNorm(out_channels,out_channels)
        self.tanh = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        #print(x.shape)
        x = self.ins(x)
        
        return self.tanh(x)


class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x + residual

class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31):
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim,2*dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        #print(x.shape)
        x = self.glu(x)
        #print(x.shape)
        x = self.depthwise_conv(x)        
        x = self.batch_norm(x)
        x = self.silu(x)
        x = self.pointwise_conv2(x)
        x = self.drop(x)
        x = x.transpose(1, 2)
        return 0.5*x + residual

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm([51,dim])
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        x, _ = self.self_attention(x, x, x)
        x = self.drop(x)
        x = x.transpose(0, 1)
        return 0.5*x + residual

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor=8, conv_kernel_size=3, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(dim, num_heads)
        self.conv = ConvModule(dim, conv_kernel_size)
        self.ff2 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.layer_norm = nn.BatchNorm1d(51)
    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.layer_norm(x)
    
class ConformerBlockL(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor=8, conv_kernel_size=3, dropout=0.1):
        super(ConformerBlockL, self).__init__()
        self.ff1 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(dim, num_heads)
        self.conv = ConvModule(dim, conv_kernel_size)
        self.ff2 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.layer_norm(x)


class PatchGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(PatchGANBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))
    

class DCNMFEncoder(nn.Module):
    def __init__(self, mel_bins=128, time_frames=51, rank=90):
        super(DCNMFEncoder, self).__init__()
        # Basis matrix W
        self.conv_w = nn.Conv2d(1, rank, kernel_size=(mel_bins, 12))
        # Activation matrix H
        self.conv_h = nn.Conv2d(1, rank, kernel_size=(1, time_frames-11))
        #self.softm = nn.Softmax(1)
        self.soft = nn.Softplus()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        W = self.conv_w(x)  # Shape: [1, rank, 1, 51]
        W = self.soft(W)#.squeeze(2)  # Shape: [1, rank, 51]
        #print(W.shape)
        
        H = self.conv_h(x)  # Shape: [1, rank, 128, 1]
        H = self.sig(H) # Shape: [1, rank, 128]
        #print(H.shape)

        return W, H
    

class Squash(nn.Module):
    """
    Squash activation function for capsule layers.
    """
    def forward(self, s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))

class PrimaryCaps(nn.Module):
    """
    Primary capsule layer for processing mel spectrograms.
    """
    def __init__(self, C, L, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.C = C
        self.L = L
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=C * L, kernel_size=kernel_size, stride=stride
        )
        self.squash = Squash()

    def forward(self, x):
        x = self.conv(x)  # Apply convolution
        H, W = x.shape[-2:]  # Get height and width of the output feature maps
        x = x.view(x.size(0), self.C, self.L, H, W).permute(0, 3, 4, 1, 2)  # Reshape to (batch, H, W, C, L)
        x = self.squash(x)  # Apply squash activation
        return x

class AttentionRouting(nn.Module):
    """
    Attention-based routing mechanism for capsule networks.
    """
    def __init__(self, input_dim, output_dim, num_capsules):
        super(AttentionRouting, self).__init__()
        self.num_capsules = num_capsules
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(1, input_dim, num_capsules * output_dim))
        self.attention_fc = nn.Linear(input_dim, num_capsules)

    def forward(self, x):
        batch_size, num_inputs, input_dim = x.size()

        # Compute predictions
        u_hat = torch.matmul(x, self.W).view(batch_size, num_inputs, self.num_capsules, self.output_dim)

        # Compute attention scores
        attention_logits = self.attention_fc(x)  # Shape: (batch_size, num_inputs, num_capsules)
        attention_weights = F.softmax(attention_logits, dim=2)

        # Compute weighted sum of predictions
        attention_weights = attention_weights.unsqueeze(-1)  # Shape: (batch_size, num_inputs, num_capsules, 1)
        s = (attention_weights * u_hat).sum(dim=1)  # Shape: (batch_size, num_capsules, output_dim)

        # Apply squash activation
        v = Squash()(s)
        return v

class DigitCaps(nn.Module):
    """
    Digit capsule layer with attention-based routing.
    """
    def __init__(self, C, L):
        super(DigitCaps, self).__init__()
        self.C = C
        self.L = L
        self.attention_routing = AttentionRouting(input_dim=L, output_dim=L, num_capsules=C)

    def forward(self, x):
        batch_size, H, W, input_C, input_L = x.size()
        x = x.view(batch_size, H * W * input_C, input_L)  # Flatten spatial and channel dimensions
        v = self.attention_routing(x)  # Apply attention-based routing
        return v

# Example usage
class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.primary_caps = PrimaryCaps(C=8, L=16, kernel_size=(9, 9), stride=(2, 2))
        self.digit_caps = DigitCaps(C=10, L=16)

    def forward(self, x):
        x = self.primary_caps(x)  # Apply primary capsule layer
        x = self.digit_caps(x)  # Apply digit capsule layer
        return x

# Input mel spectrogram of size (batch_size, 1, 128, 51)
# x = torch.randn(32, 1, 128, 51)
# model = CapsuleNetwork()
# output = model(x)
# print(output.shape)

class ConvAttention(nn.Module):
    """
    Vision Transformer-style multi-head self-attention 
    with 1x1 convolutions for Q, K, V.
    """
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 1×1 conv projections for Q, K, V
        self.q_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Optional final projection (1×1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, embed_dim, H, W)
        """
        B, C, H, W = x.shape
        
        # Generate Q, K, V using 1x1 convolutions
        Q = self.q_conv(x)  # (B, embed_dim, H, W)
        K = self.k_conv(x)  # (B, embed_dim, H, W)
        V = self.v_conv(x)  # (B, embed_dim, H, W)
        
        # Reshape for multi-head attention
        # => (B, num_heads, head_dim, H, W)
        Q = Q.view(B, self.num_heads, self.head_dim, H, W)
        K = K.view(B, self.num_heads, self.head_dim, H, W)
        V = V.view(B, self.num_heads, self.head_dim, H, W)
        
        # Flatten spatial dimension (H*W) for Q, K, V
        # => (B, num_heads, head_dim, H*W)
        Q = Q.flatten(start_dim=3)  # (B, nHeads, head_dim, HW)
        K = K.flatten(start_dim=3)  # (B, nHeads, head_dim, HW)
        V = V.flatten(start_dim=3)  # (B, nHeads, head_dim, HW)
        
        # For matrix multiplication, rearrange Q, K so:
        # Q => (B, nHeads, HW, head_dim)
        # K => (B, nHeads, head_dim, HW)
        Q = Q.permute(0, 1, 3, 2)  # (B, nHeads, HW, head_dim)
        # K stays at (B, nHeads, head_dim, HW)
        # V => we'll rearrange later, or we can do the same:
        V = V.permute(0, 1, 3, 2)  # (B, nHeads, HW, head_dim)
        
        # Scaled dot-product attention
        # => attn_weights: (B, nHeads, HW, HW)
        attn_weights = torch.matmul(Q, K) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Multiply attention weights by V
        # => out: (B, nHeads, HW, head_dim)
        out = torch.matmul(attn_weights, V)  # (B, nHeads, HW, head_dim)
        
        # Reshape out back to (B, embed_dim, H, W)
        # First permute: (B, nHeads, head_dim, HW)
        out = out.permute(0, 1, 3, 2)  # (B, nHeads, head_dim, HW)
        # Merge heads
        out = out.view(B, self.num_heads * self.head_dim, H, W)
        
        # Final 1×1 projection
        out = self.out_proj(out)  # (B, embed_dim, H, W)
        
        return out