import argparse
import os
import random

import torch
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='Make the labeled-unlabeled split')
parser.add_argument('--setting_dir', default='./data_folder/imagenet10k', metavar='DIR',
                    help='Path to save N split benchmark')
parser.add_argument('--split', default=20, type=int, metavar='N', help='Number of time steps')
parser.add_argument('--label_rate', default=0.01, type=float, help='Label rate per time step')
args = parser.parse_args()



split = args.split
datadir = args.setting_dir
label_rate = args.label_rate



for task in range(split):
    taskdir = f'{datadir}/{args.split}split/{task}/train'
    dataset = datasets.ImageFolder(taskdir) # make sure each folders can be processed by ImageFolder
    labeled = []
    index_cls = {}
    for i, t in enumerate(dataset.targets):
        if t not in index_cls.keys():
            index_cls[t] = []
        index_cls[t].append(i)
    for c in index_cls.keys():
        cls_idx = index_cls[c]
        cls_size = len(cls_idx)
        random.shuffle(cls_idx)
        labeled.extend(cls_idx[:int(cls_size * label_rate)])
    size = len(dataset.targets)
    print(f'task {task} total length {size}')
    print(f"task {task} labeled length {len(labeled)}")


    # save the index of labeled sampels and unlabeled samples
    unlabeled = torch.tensor(list(set(range(size)) - set(labeled)), dtype=torch.long)
    labeled = torch.tensor(labeled, dtype=torch.long)


    torch.save(labeled, f'{datadir}/{args.split}split/{task}/label_index-{label_rate}.torchSave')
    torch.save(unlabeled, f'{datadir}/{args.split}split/{task}/unlabeled_index-{label_rate}.torchSave')
