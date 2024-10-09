import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
from Swin_Transformer import *
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import config, Dataset, collate_fn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 200)
        print("resnet50")
    if model_name == 'swin_net':
        # net = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=pretrained)
        net = SwinNet(512, 200)
        net.load_pre('./swin_base_patch4_window12_384_22k.pth')
        print("swin_net")


    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws

class Bilinear_Pooling(nn.Module):
    def __init__(self,  **kwargs):
        super(Bilinear_Pooling, self).__init__()

    def forward(self, feature_map1, feature_map2):
        N, D1, H, W = feature_map1.size()
        feature_map1 = torch.reshape(feature_map1, (N, D1, H * W))
        N, D2, H, W = feature_map2.size()
        feature_map2 = torch.reshape(feature_map2, (N, D2, H * W))
        X = torch.bmm(feature_map1, torch.transpose(feature_map2, 1, 2)) / (H * W)
        X = torch.reshape(X, (N, D1 * D2))
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        bilinear_features = 100 * torch.nn.functional.normalize(X)
        return bilinear_features

def co_att(feature1, feature2):
    b, c, w1, h1 = feature1.shape
    # _, _, w2, h2 = fm2.shape
    fm1 = feature1.view(b, c, -1)  # B*C*S
    fm2 = feature2.view(b, c, -1)  # B*C*M
    #
    # fm1_t = fm1.permute(0, 2, 1)  # B*S*C
    #
    # # may not need to normalize
    # fm1_t_norm = F.normalize(fm1_t, dim=-1)
    # fm2_norm = F.normalize(fm2, dim=1)
    M = -1 * torch.bmm(fm1, fm2.permute(0, 2, 1))  # B*S*M
    I = F.softmax(M, dim=1)
    # M_2 = F.softmax(M.permute(0, 2, 1), dim=1)
    # new_fm2 = torch.bmm(fm1, M_1).view(b, c, w2, h2)
    Y = torch.bmm(I, fm1).view(b, c, w1, h1)

    # B, N, W, H = feature1.shape
    # x1 = feature1.reshape(B, N, W*H)
    # x2 = feature2.reshape(B, N, W*H)
    # I = -torch.bmm(x1, x2.permute(0, 2, 1))
    # I = F.softmax(I, 2)
    # Y = torch.bmm(I, fm1.reshape(B, N, W*H)).reshape(B, N, W, H)
    Y = Y + feature1
    return Y


# 计算输入特征张量的 Class Activation Map (CAM),通过计算特征均值，为特征映射中每个位置加权。最后生成一个与输入形状相同的 CAM 张量。
def get_cam(feature):
    # 获取输入特征张量的形状
    b, c, w, h = feature.shape

    # 构造一个与输入特征张量形状相同的零张量
    mask = torch.zeros([b, w, h]).cuda()

    # 遍历输入特征张量中的每个批次
    for index in range(b):
        # 获取当前批次的特征映射
        feature_maps = feature[index]
        # print(mask.shape)  torch.Size([2, 14, 14])

        # 计算特征映射的均值，作为加权因子
        weights = torch.mean(torch.mean(feature_maps, dim=-2, keepdim=True), dim=-1, keepdim=True)
        # print(weights.shape)torch.Size([2048, 1, 1])

        # 更简洁的计算 CAM 张量方法
        mask[index, :, :] = torch.mean(weights * feature_maps, dim=0)

    return mask

# 用于生成高斯归一化后的特征张量。
def get_gauss(feature):
    b, w, h = feature.shape
    # print(feature.shape)

    # 创建一个和输入张量相同形状的全0张量，并使用cuda加速
    mask = torch.zeros_like(feature).cuda()
    # print(mask.shape)

    # 遍历每一个batch，计算每个batch的均值和方差，并将所有张量标准化到0均值和1标准差
    for index in range(b):
        mean_b = torch.mean(feature[index])
        std_b = torch.std(feature[index])
        mask[index, :, :] = (feature[index] - mean_b) / std_b

    # 返回标准化后的张量
    return mask

# 定义一个特征损失函数，接受两个特征张量fs和ft作为输入
def FeatureLoss(fs, ft):
    # 获取输入特征张量fs的掩盖张量
    cam_s = get_cam(fs)

    # 对掩盖张量进行标准化处理，生成标准掩盖张量（即掩盖结果张量的标准化）
    gauss_s = get_gauss(cam_s)

    # 剪切标准化后的张量，将所有值低于0的值置为0
    mask_s = torch.clamp(gauss_s, 0)

    # 获取输入特征张量ft的掩盖张量
    cam_t = get_cam(ft)

    # 对掩盖张量进行标准化处理，生成标准掩盖张量
    gauss_t = get_gauss(cam_t)

    # 获取标准掩盖张量的绝对值，生成绝对值掩盖张量
    mask_t = torch.clamp(gauss_t, 0)

    # 计算掩盖张量之间的点积，并使用点积结果计算均方误差
    # 将均方误差作为损失并返回
    loss = torch.mean(mask_s * mask_t)
    # print('loss', loss.data)
    return loss

if __name__ == '__main__':
    x = torch.randn((2, 2048, 14, 14))
    loss = FeatureLoss(x, x)
    print('loss', loss.data)
    # print(new_layers)
