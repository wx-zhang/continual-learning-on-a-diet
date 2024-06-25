import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

from dataset.utils import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class CGLMPerTASK(Dataset):
    def __init__(self, data_path, csv_file, task, is_train=False, transform=None, timefile=None):

        self.data = pd.read_csv(csv_file, header=None, names=[
                                "class_id", "image_path", 'time'])
        if not is_train:
            self.get_test_data(task, timefile)
        self.transform = transform
        self.data_path = data_path

    def get_test_data(self, task, timefile):
        with open(timefile) as file:
            lines = file.readlines()
            task_time = float(lines[task].strip())
        self.data = self.data[self.data.iloc[:, 2] <= task_time]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        class_id = int(self.data.iloc[idx, 0])

        image_path = self.data.iloc[idx, 1]

        load_path = f"{self.data_path}/{image_path}"
        image = default_loader(load_path)
        if self.transform:
            image = self.transform(image)
        return image, class_id


class CGLM(object):
    def __init__(self, args):
        self.path = f"{args.data_path}/{args.split}split_{args.label_rate}label"
        self.data_root = args.data_root
        self.time_file = f"{args.split}time.txt"
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

    def get_total_seen_classes(self, task):
        return 10788

    def get_new_classes(self, task):
        if task == 0:
            return 10788
        else:
            return 0

    def get_labeled_set(self, task):
        train_file = f"{self.path}/{task}_labeled.csv"
        return CGLMPerTASK(self.data_root, train_file, task, is_train=True, transform=self.train_transform)

    def get_unlabeled_set(self, task):
        train_file = f"{self.path}/{task}_unlabeled.csv"
        return CGLMPerTASK(self.data_root, train_file, task, is_train=True, transform=self.unlabeled_transform)

    def get_eval_set(self, task, per_task_eval):
        assert per_task_eval == False
        test_file = f"{self.path}/test.csv"
        time_file = f"{self.path}/{self.time_file}"
        return CGLMPerTASK(self.data_root, test_file, task, is_train=False, transform=self.val_transform, timefile=time_file)
