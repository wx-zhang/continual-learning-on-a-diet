import argparse
import os
import csv
import random

parser = argparse.ArgumentParser(description='Get CGLM benchmark')
parser.add_argument('--root', default='data_folder/glm', metavar='DIR', help='path to dataset')
parser.add_argument('--benchmark', default='data_folder/cglm', metavar='DIR', help='Path to save benchmark')
parser.add_argument('--split', default=20, type=int, help='Number of splits')
parser.add_argument('--label_rate', default=0.05, type=float, help='Label rate')
args = parser.parse_args()


outpath = os.path.join(args.benchmark, f"{args.split}split_{args.label_rate}label")
os.makedirs(outpath, exist_ok=True)

with open(os.path.join(args.root, 'train.txt'), 'r') as f:
    cglm = f.readlines()

traindata = []
cls_count = 0
classes = {}

# process train set and append class index
for item in cglm:
    info = item.strip().split('\t')
    if info[0] not in classes:
        classes[info[0]] = str(cls_count)
        cls_count += 1
    info.append(classes[info[0]])
    traindata.append(info)

with open(os.path.join(args.root, 'test.txt'), 'r') as f:
    cglm = f.readlines()

# process test set and append class index
testdata = []
for item in cglm:
    info = item.strip().split('\t')
    info[0] = classes[info[0]]
    testdata.append(info)
with open(f'{outpath}/test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for data in testdata:
        writer.writerow(data)

# split train set into 20 splits
task_length = len(traindata) // args.split
taski = [traindata[task_length * i:task_length * (i + 1)] for i in range(args.split)]
# each split ending time
with open(f'{outpath}/{args.split}time.txt', 'w') as t:
    for i in range(args.split):
        t.write(taski[i][-1][-2])
        t.write('\n')

cls_count = 0
for taskid, task in enumerate(taski):
    dict = {}
    for [cls, path, time, id] in task:
        if cls not in dict.keys():
            dict[cls] = [[path, time, id]]
        else:
            dict[cls].append([path, time, id])
    with open(f'{outpath}/{taskid}_labeled.csv', 'w', newline='') as l, open(f'{outpath}/{taskid}_unlabeled.csv', 'w', newline='') as u:
        lwriter = csv.writer(l)
        uwriter = csv.writer(u)
        for cls in dict.keys():
            cur_total_length = len(dict[cls])
            cur_labeled_length = round(cur_total_length * args.label_rate)
            labeled_index = random.sample(range(cur_total_length), cur_labeled_length)
            for i in range(cur_total_length):
                path, time, id = dict[cls][i]
                if i in labeled_index:
                    lwriter.writerow([id, path, time])
                else:
                    uwriter.writerow([id, path, time])
