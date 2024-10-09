import os
from PIL import Image
import torchvision.transforms as transforms

# 读取原始图片
# img = Image.open('image (1).JPG')
img_path = r'.\pic\4.JPG'  # 假设原始图片的路径为test.jpg
img = Image.open(img_path)
save_dir =r'.\pic'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 定义多种数据增强操作
train_transform1 = transforms.Compose([
    transforms.Scale((550, 550)),
])
train_transform2 = transforms.Compose([
    transforms.Scale((550, 550)),
    transforms.RandomCrop(448, padding=8),
])

train_transform3 = transforms.Compose([
    transforms.Scale((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转

])
#
# train_transform4 = transforms.Compose([
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,  hue=0.1),  # 随机图像属性
# ])
train_transform5 = transforms.Compose([
    transforms.Scale((550, 550)),
    transforms.CenterCrop(448),
])
# train_transform6 = transforms.Compose([
#     transforms.RandomPerspective(),  # 随机透视变换
# ])
# train_transform7 = transforms.Compose([
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除
# ])

for i in range(5):
    img1 = train_transform1(img)
    img1.save(os.path.join(save_dir, f'transform1.jpg'))

    img2 = train_transform2(img)
    img2.save(os.path.join(save_dir, f'transform2.jpg'))

    img3 = train_transform3(img)
    img3.save(os.path.join(save_dir, f'transform3.jpg'))

    # img4 = train_transform4(img)
    # img4.save(os.path.join(save_dir, f'test_{i}.jpg'))

    img5 = train_transform5(img)
    img5.save(os.path.join(save_dir, f'transform5.jpg'))

    # img6 = train_transform6(img)
    # img6.save(os.path.join(save_dir, f'test_{i}.jpg'))

    # img7 = train_transform7(img)
    # img7.save(os.path.join(save_dir, '7', f'test_{i}.jpg'))