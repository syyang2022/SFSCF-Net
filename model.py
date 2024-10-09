import torch.nn as nn
import torch
from Resnet import *
import torch.nn.functional as F


class FSBM(nn.Module):
    def __init__(self, in_channel, k):
        super(FSBM, self).__init__()
        self.k = k
        self.stripconv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, fm):
        b, c, w, h = fm.shape
        print(fm.shape)
        fms = torch.split(fm, w // self.k, dim=2)
        fms_conv = map(self.stripconv, fms)
        fms_pool = list(map(self.avgpool, fms_conv))
        fms_pool = torch.cat(fms_pool, dim=2)
        fms_softmax = torch.softmax(fms_pool, dim=2)  # every parts has one score [B*C*K*1]
        fms_softmax_boost = torch.repeat_interleave(fms_softmax, w // self.k, dim=2)
        print(fms_softmax_boost.shape)
        alpha = 0.5
        fms_boost = fm + alpha * (fm * fms_softmax_boost)

        return fms_boost


class TopkPool(nn.Module):
    def __init__(self):
        super(TopkPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        topkv, _ = x.topk(5, dim=-1)
        return topkv.mean(dim=-1)


class FDM(nn.Module):
    def __init__(self):
        super(FDM, self).__init__()
        self.factor = round(1.0 / (28 * 28), 3)

    def forward(self, fm1, fm2):
        b, c, w1, h1 = fm1.shape
        _, _, w2, h2 = fm2.shape
        fm1 = fm1.view(b, c, -1)  # B*C*S
        fm2 = fm2.view(b, c, -1)  # B*C*M

        fm1_t = fm1.permute(0, 2, 1)  # B*S*C

        # may not need to normalize
        fm1_t_norm = F.normalize(fm1_t, dim=-1)
        fm2_norm = F.normalize(fm2, dim=1)
        M = -1 * torch.bmm(fm1_t_norm, fm2_norm)  # B*S*M

        M_1 = F.softmax(M, dim=1)
        M_2 = F.softmax(M.permute(0, 2, 1), dim=1)
        new_fm2 = torch.bmm(fm1, M_1).view(b, c, w2, h2)
        new_fm1 = torch.bmm(fm2, M_2).view(b, c, w1, h1)

        return self.factor * new_fm1, self.factor * new_fm2

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class fuse_enhance(nn.Module):
    def __init__(self, infeature):
        super(fuse_enhance, self).__init__()
        self.depth_channel_attention = ChannelAttention(infeature)
        self.rgb_channel_attention = ChannelAttention(infeature)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.depth_spatial_attention = SpatialAttention()

    def forward(self, r, d):
        assert r.shape == d.shape,      "rgb and depth should have same size"

        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        d_f = d * sa
        r_ca = self.rgb_channel_attention(r_f)
        d_ca = self.depth_channel_attention(d_f)

        r_out = r * r_ca
        d_out = d * d_ca
        return r_out, d_out


class PMG(nn.Module):
    def __init__(self, model, feature_size=512, class_num=200):
        super(PMG, self).__init__()

        self.features = model
        self.num_ftrs = 2048
        part_feature = 1024
        self.fuse_enhance1 = fuse_enhance(1024)
        # self.fuse_enhance2 = fuse_enhance(1024)
        # self.fuse_enhance3 = fuse_enhance(2048)

        self.pool = TopkPool()
        self.inter = FDM()

        # 通道数变为512在变成1024
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(feature_size, class_num),
        )

    def forward(self, x, y):
        _, _, r1, r2, r3 = self.features(x)
        _, _, d1, d2, d3 = self.features(y)
        # print(xf3.shape, xf4.shape, xf5.shape)

        # att1 = self.conv_block1(r1)
        # att2 = self.conv_block2(r2)
        # att3 = self.conv_block3(r3)
        # # att = att3
        # # print(att1.shape, att2.shape, att3.shape)
        # # torch.Size([2, 1024, 56, 56]) torch.Size([2, 1024, 28, 28]) torch.Size([2, 1024, 14, 14])
        #
        # btt1 = self.conv_block1(d1)
        # btt2 = self.conv_block2(d2)
        # btt3 = self.conv_block3(d3)
        #
        # _, fr2_r1 = self.inter(att1, att2)
        # rc2 = att2 + fr2_r1
        # _, fr3_r12 = self.inter(rc2, att3)
        # fm1 = att3 + fr3_r12
        #
        # _, fe2_e1 = self.inter(btt1, btt2)
        # dc2 = btt2 + fe2_e1
        # _, fe3_e12 = self.inter(dc2, btt3)
        # fm2 = btt3 + fe3_e12

        x1 = self.pool(r3)
        y1 = self.pool(d3)

        out1 = self.classifier(x1)
        out2 = self.classifier(y1)

        fr1, fd1 = self.fuse_enhance1(fm1, fm2)

        features1_self = torch.mul(fr1, btt3) + att3
        features1_other = torch.mul(fd1, att3) + att3

        features2_self = torch.mul(fd1, btt3) + btt3
        features2_other = torch.mul(fr1, btt3) + btt3

        # print(features1_self.shape, features1_other.shape)
        # torch.Size([2, 1024, 12, 12]) torch.Size([2, 1024, 12, 12])

        logit1_self = F.adaptive_avg_pool2d(features1_self, (1, 1))
        logit1_self = logit1_self.flatten(1)
        out3 = self.classifier(logit1_self)

        logit1_other = F.adaptive_avg_pool2d(features1_other, (1, 1))
        logit1_other = logit1_other.flatten(1)
        out4 = self.classifier(logit1_other)

        logit2_self = F.adaptive_avg_pool2d(features2_self, (1, 1))
        logit2_self = logit2_self.flatten(1)
        out5 = self.classifier(logit2_self)

        logit2_other = F.adaptive_avg_pool2d(features2_other, (1, 1))
        logit2_other = logit2_other.flatten(1)
        out6 = self.classifier(logit2_other)

        return out1, out2, out3, out4, out5, out6, features1_self, features1_other, features2_self, features2_other


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


if __name__ == '__main__':
    x = torch.randn((2, 3, 448, 448))
    y = torch.randn((2, 3, 448, 448))
    net = resnet50(pretrained=True)
    model = PMG(net, 512, 200)
    out1, out2, out3, out4, out5, out6, features1_self, features1_other, features2_self, features2_other = model(x, y)

    # print(xc1.shape, xc2.shape, xc3.shape, x_concat.shape)