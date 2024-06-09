import os
import argparse
from imagenet1k import imagenet1k


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError:
        pass


parser = argparse.ArgumentParser(description='Get ImageNet benchmark')
parser.add_argument('--root21k', default='data_folder/imagenet',
                    metavar='DIR', help='ImageNet 21k dataroot')
parser.add_argument('--base_store_root', default='data_folder',
                    metavar='DIR', help='ImageNet 21k dataroot')
parser.add_argument('--benchmark_name', default='imagenet10k',
                    metavar='DIR', help='Path to save benchmark')

args = parser.parse_args()


abs_path = os.path.abspath("./")
data_root = os.path.join(abs_path, args.base_store_root, args.benchmark_name)
i21k_root = os.path.join(abs_path, args.root21k)
os.makedirs(f'{data_root}/train', exist_ok=True)
os.makedirs(f'{data_root}/val', exist_ok=True)

i1k_cls = set(imagenet1k)
i21k_cls = set(os.listdir(f'{i21k_root}/train'))

# Get unique classes
classes = i21k_cls - i1k_cls


print(f"Link classes from {i21k_root} to {data_root}")

for c in classes:
    symlink_force(f'{i21k_root}/train/{c}', f'{data_root}/train/{c}')
    symlink_force(f'{i21k_root}/val/{c}', f'{data_root}/val/{c}')
