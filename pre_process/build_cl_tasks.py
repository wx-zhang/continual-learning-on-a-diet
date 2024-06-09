import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser(
    description='Split benchmark into N time steps')
parser.add_argument('--datadir', default='data_folder/imagenet10k', metavar='DIR',
                    help='Path to save non-overlap  benchmark')
parser.add_argument('--split', default=20, type=int,
                    metavar='N', help='Number of time steps')
args = parser.parse_args()

# check if the datadir is absolute path
if not os.path.isabs(args.datadir):
    args.datadir = os.path.abspath(args.datadir)

split = args.split
setting = f"{args.datadir}/{split}split"
data_root = args.datadir

os.makedirs(setting, exist_ok=True)

# split data and link them into setting dir
classes = os.listdir(f'{data_root}/train')
random.shuffle(classes)
classes_per_split = np.array_split(classes, split)

print(f"Classes per split: {len(classes_per_split[0])}")

for task in range(split):
    os.makedirs(f'{setting}/{task}', exist_ok=True)
    os.makedirs(f'{setting}/{task}/val', exist_ok=True)
    os.makedirs(f'{setting}/{task}/train', exist_ok=True)

    for c in classes_per_split[task]:
        os.symlink(f'{data_root}/val/{c}', f'{setting}/{task}/val/{c}')
        os.symlink(f'{data_root}/train/{c}', f'{setting}/{task}/train/{c}')
