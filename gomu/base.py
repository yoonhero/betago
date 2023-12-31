import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from copy import deepcopy

# Using binary characteristic for getting max divisable time by 2.
def n_divide_by_2(x):
    return str(bin(x))[2:][::-1].index("1")

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # rank reduction for feature extraction(?)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # [b, c]
        y = self.fc(y).view(b, c, 1, 1) # [b, c, 1, 1]
        return x * y

class MBConv(nn.Module):
    def __init__(self, inp, oup, downsample=False, **kwargs):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        hidden_dim = int(inp * 3)

        if self.downsample:
            # Make smaller for resid-con
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        else:
            self.proj = nn.ConvTranspose2d(inp, oup, 2, 2, 0, bias=False)
        
        if self.downsample:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.resid_proj = nn.Conv2d(inp*2, inp, 1, 1, 0, bias=False)
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2, 0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                nn.ConvTranspose2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x, xt=None):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        
        x = torch.cat((x, xt), dim=1)
        x = self.resid_proj(x)
        return self.proj(x) + self.conv(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # projection disablinng when casual attention
        projection = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # paramter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2*self.ih-1)*(2*self.iw-1), heads),
        )

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relaitve_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relaitve_index)

        # b c ih iw
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        
        self.out_proj = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout),
        ) if projection else nn.Identity()

    def forward(self, x):
        # multihead attention implementation
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> b h n d", h=self.heads
        ), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # User "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = rearrange(
            relative_bias, "(h w) c -> 1 c h w", h=self.ih*self.iw, w=self.ih*self.iw
        )
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        self.ih, self.iw = image_size
        self.downsample = downsample

        if downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        # self.attn = nn.Sequential(
        #     Rearrange("b c ih iw -> b (ih iw) c"),
        #     PreNorm(inp, self.attn, nn.LayerNorm),
        #     Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw)
        # )

        self.ff = nn.Sequential(
            Rearrange("b c ih iw -> b (ih iw) c"),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        # if self.downsample:
            # x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        # else:
            # x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class Unet(nn.Module):
    def __init__(self, nrow, ncol, channels):
        super().__init__()
        self.nrow = nrow
        self.ncol = ncol

        # Get a max down time for avoiding dimension error problem.
        self.max_down = min(n_divide_by_2(self.ncol), n_divide_by_2(self.nrow))
        assert (len(channels)-1) <= self.max_down, "Please check channel."

        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for inp, oup in zip(channels, channels[1:]):
            self.downsample.append(MBConv(inp=inp, oup=oup, downsample=True))
        
        for inp, oup in zip(channels[::-1], channels[::-1][1:]):
            self.upsample.append(MBConv(inp=inp, oup=oup, downsample=False))

        self.latent_transformer = Transformer(channels[-1], channels[-1], (5, 5))
        # self.outc = nn.Conv2d(channels[0], channels[0], 1)

    def forward(self, x):
        tmp = []
        for block in self.downsample:
            x = block(x)
            tmp.append(x)
        
        x = self.latent_transformer(x)
        
        for i, block in enumerate(self.upsample):
            x = block(x, tmp[-1-i])
        
        # x = self.outc(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, inp, mid, oup, stride=1, dropout=0):
        super().__init__()

        self.res_block = nn.Sequential(
            nn.Conv2d(inp, mid, 3, stride, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
            nn.Conv2d(mid, oup, 3, 1, 1, bias=False),
            nn.BatchNorm2d(oup)
        )
        self.activation = nn.ReLU()

        self.downsample = inp != oup or stride != 1
        if self.downsample:
            self.down = nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.activation_block = nn.Conv2d(inp, oup, 1, 1, 0, bias=True)

        self.dropout = nn.Dropout2d(p=dropout)
    
    def forward(self, x):
        residual = x
        out = self.res_block(x)
        if self.downsample:
            residual = self.down(residual)
        else:
            residual = self.activation_block(residual)
        out = out + residual
        out = self.activation(out)
        return out

class PolicyValueNet(nn.Module):
    def __init__(self, nrow, ncol, channels, dropout):
        super().__init__()
        self.nrow = nrow
        self.ncol = ncol

        self.convs = nn.ModuleList()

        for inp, oup in zip(channels, channels[1:]):
            # => half
            hidden_dim = inp * 3

            # conv = ResBlock(inp, hidden_dim, inp)
            conv = ResBlock(inp, hidden_dim, oup, dropout=dropout)
            self.convs.append(conv)

            #downsample = ResBlock(inp, hidden_dim, oup, stride=2)
            #self.convs.append(downsample)

        # #2. Without downsampling??
        # scale = 2**(len(channels) - 1)
        # final_row, final_col = self.nrow / scale, self.ncol / scale
        # final = int(final_row * final_col * channels[-1])
        final = nrow * ncol * channels[-1]
        self.conv1x = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 1, 1, 0),
        )
        self.ff = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            # nn.Linear(final, final),
            nn.Linear(final, final*3),
            nn.GELU(),
            nn.Linear(final*3, final), 
            nn.GELU(),
            nn.Linear(final, 1),
            nn.Sigmoid()
        )

        self._intialize_weights()
    
    def _intialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # tmp = deepcopy(x)
        for conv in self.convs:
            x = conv(x)
        
        policy = self.conv1x(x)
        value = self.ff(x)

        # Masking was bad.
        # if self.train:
        #     # B = tmp.shape[0]
        #     not_free_space = tmp.sum(1).unsqueeze(1)
        #     policy = policy.masked_fill(not_free_space==1, -1e9)

        return policy, value


def get_total_parameters(net):
    return sum([i.nelement() for i in net.parameters()])


if __name__ == "__main__":
    # net = Unet(20, 20, [2, 5, 10])
    channels = [2, 64, 128, 256, 128, 64, 32, 1]
    net = PolicyValueNet(20, 20, channels, 0)
    # print(net/.total())
    to_feed = torch.zeros((10, 2, 20, 20))
    # print(net(to_feed))
    print(net(to_feed))
