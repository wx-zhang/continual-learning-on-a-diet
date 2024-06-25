import time
from trainer.pretrain import PretrainTrainer
from data import get_buffer, get_loader, get_and_update_buffer
from utils import wrap_model
from trainer.utils import *


class DietTrainer(PretrainTrainer):
    def train(self, task, model, dataset):
        total_classes = dataset.get_total_seen_classes(task)
        self.mixup(total_classes)
        print ('Train stage 1: mixed training')
        self.train_stage1(task,model,dataset)
        print ('Train stage 2: train with buffer')
        self.train_stage2(task,model,dataset)

    def compute_batch_size(self, task):
        """
        compute stage 1 batch size
        return unlabeled, labeled, buffer batch size
        """
        assert (self.args.sampling in ['batchmix', 'uniform']) or self.args.replay_buffer_size == 0
        if self.args.replay_buffer_size == 0:
            unlabeled_batch_size = int(self.args.batch_size * self.args.unlabeled_data_ratio)
            labeled_batch_size = self.args.batch_size - unlabeled_batch_size
            batch_sizes = [unlabeled_batch_size, labeled_batch_size, 0]
        elif self.args.sampling == 'uniform':
            unlabeled_batch_size = int(self.args.batch_size * self.args.unlabeled_data_ratio)
            labeled_batch_size = self.args.batch_size - unlabeled_batch_size
            batch_sizes = [unlabeled_batch_size, 0, labeled_batch_size]
        else:
            unlabeled_batch_size = int(self.args.batch_size * self.args.unlabeled_data_ratio)
            labeled_batch_size = self.args.batch_size - unlabeled_batch_size
            batch_sizes = [unlabeled_batch_size, labeled_batch_size // 2, labeled_batch_size// 2]
        print(f"Task {task} stage 1, unlabeled batch size {batch_sizes[0]}, labeled batch size {batch_sizes[1]}, buffer batch size {batch_sizes[2]}")
        return batch_sizes

    def get_stage1_loader(self, task, dataset):

        dataloaders = {}
        batch_sizes = self.compute_batch_size(task)



        print (f"Task {task}, loading unlabeled dataset for stage 1...")
        unlabeled_set = dataset.get_unlabeled_set(task)
        unlabeled_loader = get_loader(unlabeled_set, batch_sizes[0], self.args)
        print (f"Task {task}, unlabeled set loaded, size {len(unlabeled_set)}")
        dataloaders['cur_unlabeled'] = unlabeled_loader

        
        print(f"Task {task}, loading labeled dataset for stage 1...")
        cur_labeled_set = dataset.get_labeled_set(task)
        print(f"Task {task}, labeled set loaded, size {len(cur_labeled_set)}")
        cur_labeled_loader = get_loader(cur_labeled_set, batch_sizes[1], self.args)
        if cur_labeled_loader is not None:
            dataloaders['cur_labeled'] = cur_labeled_loader

        print(f"Task {task}, checking buffer for stage 1...")
        buffer_set = get_and_update_buffer(task, dataset, self.args)
        if buffer_set is not None and batch_sizes[2] > 0:
            print(f"Task {task}, buffer set loaded, size {len(buffer_set)}")
            buffer_loader = get_loader(buffer_set, batch_sizes[2], self.args)
            dataloaders['buffer'] = buffer_loader
        else:
            print(f"Task {task}, no buffer")
        return dataloaders

    def get_stage2_loader(self,task,dataset):
        dataloaders = {}

        buffer_set = get_buffer( dataset, self.args)
        print(f"Task {task}, buffer set loaded, size {len(buffer_set)}")
        buffer_loader = get_loader(buffer_set, self.args.batch_size, self.args)
        dataloaders['buffer'] = buffer_loader


        return dataloaders

    def train_stage2(self, task, model, dataset):
        start_time = time.time()

        # init optimizer
        blr = compute_lr(self.args) * self.args.train_stage2_lr_scale
        optimizer = self.get_optimizer(model, blr)

        # init dataloader
        train_loader = self.get_stage2_loader(task, dataset)

        # compute iterations
        stage2_iterations = self.compute_iter(stage=2)
        if stage2_iterations == 0:
            return

        # rewrap the model
        model = wrap_model(model, True, self.args)

        # train
        self.train_iterations(model, optimizer, train_loader, task, stage2_iterations, blr)

        # log overall training time
        total_time = all_reduce_mean(time.time() - start_time, self.args.rank, self.args.world_size)
        if self.args.rank == 0 and self.args.wandb.enable:
            wandb_log(f'{task}/total_finetune_time', total_time)

    
