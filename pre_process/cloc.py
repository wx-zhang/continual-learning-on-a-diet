import os
import csv
import random
import torch
import argparse

parser = argparse.ArgumentParser(description='Get CGLM benchmark')
parser.add_argument('--root', default='data_folder/cloc',
                    metavar='DIR', help='path to dataset')
parser.add_argument('--benchmark', default='data_folder/cloc_split',
                    metavar='DIR', help='Path to save benchmark')
parser.add_argument('--split', default=20, type=int, help='Number of splits')
parser.add_argument('--label_rate', default=0.005,
                    type=float, help='Label rate')
args = parser.parse_args()

data_path = args.benchmark
label_rate = 0.005
split = args.split
outpath = os.path.join(data_path, f'{split}_{label_rate}')
os.makedirs(outpath, exist_ok=True)

data_root = args.root
label_root = os.path.join(data_root, 'train_labels.torchSave')
labels = torch.load(label_root)


task_length = len(labels) // split
taski = [labels[task_length * i:task_length * (i + 1)] for i in range(split)]
end_indices = [i*task_length-1 for i in range(1, split+1)]

time_root = os.path.join(data_root, 'train_time.torchSave')
time = torch.load(time_root)
indices_time = [time[i] for i in end_indices]
torch.save(indices_time, f'{outpath}/time.torchSave')
print('time file saved...')

for taskid, task in enumerate(taski):
    dict = {}
    for index, cls in enumerate(task):
        if cls not in dict.keys():
            dict[cls] = [index]
        else:
            dict[cls].append(index)

    labeled_indices = []
    unlabeled_indices = []
    for cls in dict.keys():
        cur_total_length = len(dict[cls])
        cur_labeled_length = round(cur_total_length * label_rate)
        labeled_index = random.sample(dict[cls], cur_labeled_length)
        unlabeled_index = list(set(dict[cls]) - set(labeled_index))
        labeled_indices.extend(labeled_index)
        unlabeled_indices.extend(unlabeled_index)
    print(f'task{taskid}, total length {len(task)}, labeled length {len(labeled_indices)}, unlabeled length {(len(unlabeled_indices))}')
    torch.save(labeled_indices, f'{outpath}/{taskid}_labeled.torchSave')
    torch.save(unlabeled_indices, f'{outpath}/{taskid}_unlabeled.torchSave')
