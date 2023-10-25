import torch

import torch.nn as nn


class CNN_CN(nn.Module):
    def __init__(self,mid_planes):
        super().__init__()
        self.conv0 = nn.Conv2d(1, mid_planes, kernel_size=3, stride=1, padding=1,padding_mode='circular', bias=False)
        self.conv1 = nn.Conv2d(mid_planes,mid_planes,kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)
        self.conv2 = nn.Conv2d(mid_planes,mid_planes,kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)
        self.conv3 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)
        self.conv4 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)
        self.conv5 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)
        self.conv6 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)

        self.convF = nn.Conv2d(mid_planes,1,kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)
    def full(self, x):
        full_left = torch.clone(x[:, :, :, [0]])
        full = torch.cat((x, full_left), 3)
        full_bottom = torch.clone(full[:, :, [-1], :])
        full = torch.cat((full_bottom, full), 2)
        return full


    def integral(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        a = x.mean([2, 3]).unsqueeze(2).unsqueeze(3)
        b = a.expand(-1, -1, h_x, w_x)
        return b


    def forward(self,x,type='ACNN'):
        residual = self.conv0(x)
        mass = self.integral(x)


        out = self.conv0(x)
        #resblock1
        out = self.conv1(out)
        out = torch.tanh(out)
        out = self.conv2(out)
        out = residual+out
        out = torch.tanh(out)
        #resblock2
        residual = out
        out = self.conv3(out)
        out = torch.tanh(out)
        out = self.conv4(out)
        out = residual + out
        out = torch.tanh(out)

        #resblock3
        residual = out
        out = self.conv5(out)
        out = torch.tanh(out)
        out = self.conv6(out)
        out = residual + out
        out = torch.tanh(out)

        out= self.convF(out)

        # Difference of mass
        if type=='ACNN':
            out = out

            # Bound limiter

            upper = torch.full_like(out, 1.001) # Case specific bound value
            low = torch.full_like(out, -1.001)
            out = torch.where(out >= 1.001, upper, out)
            out = torch.where(out <= -1.001, low, out)
        elif type == 'mACNN':
            out1 = self.integral(out) - mass
            out = out - out1

            upper = torch.full_like(out, 1.1647) # Case specific bound value
            low = torch.full_like(out, -1.1647)
            out = torch.where(out >= 1.1647, upper, out)
            out = torch.where(out <= -1.1647, low, out)





        return out











