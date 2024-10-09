from __future__ import print_function
import os
from PIL import Image
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset import config, Dataset, collate_fn
import logging
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import config, Dataset, collate_fn
from model import *
from torch.utils.tensorboard import SummaryWriter
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

log_time = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
log_name = str(200) + log_time + 'bird'
log_dir = os.path.join(os.getcwd(), 'logs', log_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    print("kind", store_name)
# Data
    print('==> Preparing data..')
    train_root, test_root, train_pd, test_pd, cls_num = config(data=store_name)
    data_transforms = transforms.Compose([
        transforms.Resize((440, 440)),
        transforms.RandomCrop(384, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # trainset = torchvision.datasets.ImageFolder(root='D:/dataset/Birds-CN-CNN/train', transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_dataset = Dataset(train_root, train_pd, train=True, transform=data_transforms, num_positive=1)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_fn)
    
    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='swin_net', pretrain=True, require_grad=True)

    netp = torch.nn.DataParallel(net)

    # GPU
    device = torch.device("cuda")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(netp.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nb_epoch, 0)
    
    max_val_acc = 0  # 记录历史最佳精度
    alpha = 1.0  # 默认设置为1
    best_model_path = None

    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0

        for batch_idx, data in enumerate(trainloader):
            idx = batch_idx
            img, positive_img, label = data
            inputs = img.to(device)
            positive_input = positive_img.to(device)
            targets = torch.Tensor(label).type(torch.int64).to(device)

            inputs, targets = Variable(inputs), Variable(targets)

            lam = np.random.beta(alpha, alpha)
            images_a, images_b = inputs, positive_input
            mixed_images = lam * images_a + (1 - lam) * images_b

            # update learning rate
            scheduler.step()

            out1, out2, out3, out4, out5, out6, features1_self, features1_other, features2_self, features2_other = netp(inputs, positive_input)
            loss1 = CELoss(out1, targets)
            loss2 = CELoss(out2, targets)
            loss3 = CELoss(out3, targets)
            loss4 = CELoss(out4, targets)
            loss5 = CELoss(out5, targets)
            loss6 = CELoss(out6, targets)
            loss7 = FeatureLoss(features1_self, features1_other)
            loss8 = FeatureLoss(features2_self, features2_other)

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out1.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            train_loss += (loss1.item() + loss2.item() + loss3.item() + loss4.item() +
                           loss5.item() + loss6.item() + loss7.item() + loss8.item())

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f\n' % (
                epoch, train_acc, train_loss))

        # Conduct testing
        val_acc_com, val_acc, val_loss = test(net, CELoss, 12, test_root, test_pd)

        # 如果当前验证精度比历史最佳精度高，保存模型并更新最佳精度
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com
            if best_model_path is not None and os.path.exists(best_model_path):
                 os.remove(best_model_path)
            best_model_path = f'./{store_name}/best_model_{max_val_acc:.4f}.pth'
            torch.save(net.state_dict(), best_model_path)

        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, best_acc = %.5f, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, max_val_acc, val_acc, val_acc_com, val_loss))

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc_com, epoch)

        # 保存当前模型
        torch.save(net.state_dict(), f'./{store_name}/current_model.pth')


def test(net, criterion, batch_size, test_root, test_pd):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Resize((510, 510)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # testset = torchvision.datasets.ImageFolder(root='./bird/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = Dataset(test_root, test_pd, train=False, transform=transform_test)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx

            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)

            mixed_images = inputs
            out1, out2, out3, out4, out5, out6, _, _, _, _ = net(inputs, mixed_images)
            loss1 = criterion(out1, targets)
            loss2 = criterion(out2, targets)
            loss3 = criterion(out3, targets)

            loss = loss1 + loss2 + loss3
            test_loss += loss.item()
            _, predicted = torch.max(out1.data, 1)
            # _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            # correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                    batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total,
                    100. * float(correct_com) / total, correct_com, total))

        test_acc = 100. * float(correct) / total
        test_acc_en = 100. * float(correct_com) / total
        test_loss = test_loss / (idx + 1)

        return test_acc, test_acc_en, test_loss

train(nb_epoch=200,             # number of epoch
         batch_size=8,         # batch size
         store_name='bird',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='')         # the saved model where you want to resume the training
