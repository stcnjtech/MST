import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    ## type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    """
    inputs: [bs, nC=28, H=256, W'=310]
    outputs: [bs, nC=28, H=256, W=256]
    """
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MaskGuidedMechanism(nn.Module):
    def __init__(
            self, n_feat):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        """
        mask_shift: [bs, nC, H, W']
        return: [bs, nC, H, W]
        """
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift) # [bs, nC, H, W']
        # [bs, nC, H, W'] --> [bs, nC, H, W'] --> [bs, nC, H, W']
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map # [bs, nC, H, W']
        mask_shift = res + mask_shift # [bs, nC, H, W']
        mask_emb = shift_back(mask_shift) # [bs, nC, H, W]
        return mask_emb

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.mm = MaskGuidedMechanism(dim)
        self.dim = dim

    def forward(self, x_in, mask=None):
        """
        x_in: [bs, H, W, nC]
        mask: [1, H, W', nC]
        return: [bs, H, W, nC]
        """
        b, h, w, c = x_in.shape # [bs, H, W, nC]
        x = x_in.reshape(b,h*w,c) # [bs, H * W, nC]
        # q_inp, k_inp, v_inp: [bs, H * W, heads * dim_head]
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # [1, H, W', nC] --> [1, nC, H, W'] --> [1, nC, H, W] --> [1, H, W, nC]
        mask_attn = self.mm(mask.permute(0,3,1,2)).permute(0,2,3,1)
        if b != 0:
            mask_attn = (mask_attn[0, :, :, :]).expand([b, h, w, c]) # [bs, H, W, nC]
        # q, k, v: [bs, H * W, heads * dim_head] --> [bs, heads, H * W, dim_head]
        # mask_attn: [bs, H, W, nC] --> [bs, H * W, nC] --> [bs, heads, H * W, dim_head]
        q, k, v, mask_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)))
        v = v * mask_attn # [bs, heads, H * W, dim_head]
        # q, k, v: [bs, heads, dim_head, H * W]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        # [bs, heads, dim_head, H * W] @ [bs, heads, H * W, dim_head] --> [bs, heads, dim_head, dim_head]
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale # [bs, heads, dim_head, dim_head]
        attn = attn.softmax(dim=-1)
        # [bs, heads, dim_head, dim_head] @ [bs, heads, dim_head, H * W] --> [bs, heads, dim_head, H * W]
        x = attn @ v # [bs, heads, dim_head, H * W]
        x = x.permute(0, 3, 1, 2) # [bs, H * W, heads, dim_head]
        x = x.reshape(b, h * w, self.num_heads * self.dim_head) # [bs, H * W, heads * dim_head]
        out_c = self.proj(x).view(b, h, w, c) # [bs, H * W, heads * dim_head] --> [bs, H * W, nC] --> [bs, H, W, nC]
        # [bs, H * W, heads * dim_head] --> [bs, H, W, nC] --> [bs, nC, H, W] ----> [bs, nC, H, W] ----> [bs, H, W, nC]
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p # [bs, H, W, nC]

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [bs, H, W, nC]
        return: [bs, H, W, nC]
        """
        # [bs, H, W, nC] --> [bs, nC, H, W] --> [bs, nC, H, W]
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1) # [bs, H, W, nC]

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mask):
        """
        x: [bs, nC, H, W]
        mask: [1, nC, H, W']
        return out: [bs, nC, H, W]
        """
        x = x.permute(0, 2, 3, 1) # [bs, H, W, nC]
        for (attn, ff) in self.blocks:
            x = attn(x, mask=mask.permute(0, 2, 3, 1)) + x # [bs, H, W, nC]
            x = ff(x) + x # [bs, H, W, nC]
        out = x.permute(0, 3, 1, 2) # [bs, nC, H, W]
        return out


class MST(nn.Module):
    def __init__(self, dim=28, stage=3, num_blocks=[2,2,2]):
        """
        stage: encoder和decoder的层数,每层包括MSAB + Conv2d * 2)
        num_blocks: encoder和decoder每层的MSAB包括的MS_MSA个数
        """
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        """
        x: [bs, nC, H, W]
        mask: [1, nC, H, W']
        return out: [bs, nC, H, W]
        """
        if mask == None:
            mask = torch.zeros((1,28,256,310)).cuda()

        # Embedding
        fea = self.lrelu(self.embedding(x)) # [bs, nC, H, W]

        # Encoder
        fea_encoder = []
        masks = []
        for (MSAB, FeaDownSample, MaskDownSample) in self.encoder_layers:
            fea = MSAB(fea, mask) # stage1:[bs, dim, H, W]; stage2:[bs, dim * 2, H//2, W//2]
            masks.append(mask)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea) # stage1:[bs, dim * 2, H//2, W//2]; stage2:[bs, dim * 4, H//4, W//4]
            mask = MaskDownSample(mask) # stage1:[1, dim * 2, H//2, W'//2]; stage2:[bs, dim * 4, H//4, W//4]

        # Bottleneck
        fea = self.bottleneck(fea, mask) # [bs, dim * 4, H//4, W//4]

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea) # stage1:[bs, dim * 2, H//2, W//2]; stage2:[bs, dim, H, W]
            # stage1:[bs, dim * 2 * 2, H//2, W//2] --> [bs, dim * 2, H//2, W//2]
            # stage2:[bs, dim * 2, H, W] --> [bs, dim, H, W]
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = LeWinBlcok(fea, mask) # stage1:[bs, dim * 2, H//2, W//2]; stage2:[bs, dim, H, W] 

        # Mapping
        out = self.mapping(fea) + x # [bs, 28, H, W]

        return out






















