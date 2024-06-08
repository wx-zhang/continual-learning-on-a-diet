import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from dataset.utils import *
TRAIN_INDEX_PATH='/ibex/ai/reference/CLOC/release/train_store_loc.torchSave'
TRAIN_LABEL_PATH='/ibex/ai/reference/CLOC/release/train_labels.torchSave'
TEST_INDEX_PATH='/ibex/ai/reference/CLOC/release/yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250_valid_files_2004To2014_compact_val.csv'
IMAGES_PATH='/ibex/ai/reference/CLOC/release/dataset/images'



class CLOCPerTask(Dataset):
    def __init__(self, data, label=None, is_train=False, transform=None, index_file=None, timefile=None):
        if is_train:
            self.get_train_data(index_file, data, label)
        else:
            self.get_test_data(timefile, data)
        self.is_train = is_train
        self.transform = transform


    def get_train_data(self, file, data, label):

        self.indices = torch.load(file)
        self.data = [data[i] for i in self.indices]
        self.label = [label[i] for i in self.indices]
        print (f'Number of train samples: {len(self.data)}')

    def get_test_data(self,  timefile, data):

        # time = torch.load(timefile)[task]
        self.data = data[self.data.iloc[:, 2] <= time]
        print (f'Number of test samples: {len(self.data)}')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        if self.is_train:
            class_id = self.label[idx]
            image_path = self.data[idx].strip()
        else:
            class_id = self.data.iloc[idx, 0]
            image_path = self.data.iloc[idx, 4]
        path = os.path.join(IMAGES_PATH, image_path)

        image = default_loader(path)
        if self.transform:
            image = self.transform(image)

        return image, class_id


class CLOC(object):
    def __init__(self, args):
        self.path = f"{args.data_path}/{args.split}_{args.label_rate}"
        self.data_root = args.data_root
        self.train_data = torch.load(f"{self.data_root}/train_store_loc.torchSave")
        self.train_label = torch.load(f"{self.data_root}/train_labels.torchSave")
        self.test_data = pd.read_csv(f"{self.data_root}/yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250_valid_files_2004To2014_compact_val.csv", header=None, names=["class_id", "date", 'time', 'cell', 'path'])
        self.time_file_path =  f'{self.path}/time.torchSave'
        self.time = torch.load(self.time_file_path)
        self.train_transform = build_transform(is_train=True, input_size=args.input_size)
        self.val_transform = build_transform(is_train=False, input_size=args.input_size)
        self.unlabeled_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
    def get_total_seen_classes(self, task):
        return 713

    def get_new_classes(self, task):
        if task == 0:
            return 713
        else:
            return 0
    def get_labeled_set(self, task):
        index_file = f"{self.path}/{task}_labeled.torchSave"
        return CLOCPerTask(self.train_data, self.train_label, is_train=True, transform=self.train_transform, index_file=index_file)
    def get_unlabeled_set(self, task):
        index_file = f"{self.path}/{task}_unlabeled.torchSave"
        return CLOCPerTask(self.train_data, self.train_label, is_train=True, transform=self.unlabeled_transform, index_file=index_file)
    def get_eval_set(self, task):
        return CLOCPerTask(self.test_data, timefile=self.time[task], transform=self.val_transform)

    




