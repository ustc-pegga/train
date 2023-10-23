import torch.nn as nn
import torch



class Conv(nn.Module):
    def __init__(self, in_c=64, out_c=64, kernel_size=3, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)

        return x

class ConvBN(nn.Module):
    def __init__(self, in_c=64, out_c=64, kernel_size=3, stride=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def conv_dw(inp, oup, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class DWConv(nn.Module):
    def __init__(self, in_c=64, out_c=64, kernel_size=3, stride=1):
        super(DWConv, self).__init__()
        self.conv_dw = nn.Conv2d(in_c, in_c, kernel_size, stride, 1, groups=in_c, bias=False)
    def forward(self, x):
        x = self.conv_dw(x)
        return x


class DWConvBN(nn.Module):
    def __init__(self, in_c=64, out_c=64, kernel_size=3, stride=1):
        super(DWConvBN, self).__init__()
        self.conv_dw = conv_dw(in_c, out_c, kernel_size, stride)
    def forward(self, x):
        x = self.conv_dw(x)
        return x

