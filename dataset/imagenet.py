import os
import torch
from torch.utils.data import Subset, ConcatDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataset.utils import *


class trans_label(object):
    """
    Label transformation for folder-based ImageNet dataset
    """

    def __init__(self, task, path):
        self.task = task

        self.pre_classes = sum(
            [len(os.listdir(f'{path}/{i}/val/')) for i in range(task)])

    def __call__(self, y):
        return y + self.pre_classes


class ImageNet(object):
    """
    ImageNet dataset class
    """

    def __init__(self, args):
        self.path = f"{args.data_path}/{args.split}split/"
        self.label_rate = args.label_rate
        self.train_transform = build_transform(
            is_train=True, input_size=args.input_size)
        self.val_transform = build_transform(
            is_train=False, input_size=args.input_size)
        self.unlabeled_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def get_new_classes(self, task):
        return len(os.listdir(f'{self.path}/{task}/val/'))

    def get_total_seen_classes(self, task):
        return sum([len(os.listdir(f'{self.path}/{i}/val/')) for i in range(task+1)])

    def get_labeled_set(self, task):
        cur_folder = f"{self.path}/{task}/train/"
        data = datasets.ImageFolder(
            cur_folder, self.train_transform, target_transform=trans_label(task, self.path))
        label_index = torch.load(
            f'{self.path}/{task}/label_index-{self.label_rate}.torchSave')
        labeled_set = Subset(data, label_index)
        return labeled_set

    def get_unlabeled_set(self, task):
        cur_folder = f"{self.path}/{task}/train/"
        data = datasets.ImageFolder(cur_folder, self.unlabeled_transform)
        unlabeled_index = torch.load(
            f'{self.path}/{task}/unlabeled_index-{self.label_rate}.torchSave')
        unlabeled_set = Subset(data, unlabeled_index)
        return unlabeled_set

    def get_eval_set(self, task, per_task_eval):
        if per_task_eval:

            cur_folder = f"{self.path}/{task}/val/"
            valset = datasets.ImageFolder(
                cur_folder, self.val_transform, target_transform=trans_label(task, self.path))
            return valset
        else:
            data = []
            for i in range(task+1):
                cur_folder = f"{self.path}/{i}/val/"
                data.append(datasets.ImageFolder(
                    cur_folder, self.val_transform, target_transform=trans_label(i, self.path)))
            return ConcatDataset(data)
