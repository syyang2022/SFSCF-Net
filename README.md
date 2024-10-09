
# SFSCF-Net
 
Code release for Significant Feature Suppression and Cross-festure Fusion Networks for Fine-grained Visual Classification
 
### Requirement
 
python 3.6

PyTorch >= 1.3.1

torchvision >= 0.4.2

### Training

1. Download datatsets for FGVC (e.g. CUB-200-2011, Standford Cars, FGVC-Aircraft, etc) and organize the structure as follows:
```
dataset
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── test
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```

2. Train from scratch with ``train.py``.


### Citation
 
Please cite our paper if you use in your work.
```
@InProceedings{
  title={Significant Feature Suppression and Cross-festure Fusion Networks for Fine-grained Visual Classification},
  author={Yang, Shengying and Yang, Xinqi and Wu, Jianfeng and Feng, Boyang},
  booktitle = {},
  year={}
}

```

## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- syyang@zust.edu.cn
- wujianfengwz1020@163.com

