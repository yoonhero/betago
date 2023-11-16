import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        x = torch.cat((x, xt), 1)
        x = self.resid_proj(x)
        return self.proj(x) + self.conv(x)


# Compare Efficiency during using concatenate unet version.
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

    def forward(self, x):
        tmp = []
        for block in self.downsample:
            x = block(x)
            tmp.append(x)
        
        for i, block in enumerate(self.upsample):
            x = block(x, tmp[-1-i])
        
        del tmp

        return x


if __name__ == "__main__":
    net = Unet(20, 20, [1, 10, 20])
    to_feed = torch.zeros((1, 1, 20, 20))
    print(net(to_feed))
