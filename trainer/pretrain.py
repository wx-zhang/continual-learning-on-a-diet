import time


from trainer.base import BaseTrainer
from trainer.utils import *
from data import get_loader, get_and_update_buffer


class PretrainTrainer(BaseTrainer):

    def train(self, task, model, dataset):

        print('Train stage 1: Pretraining')
        self.train_stage1(task, model, dataset)
        print('Train stage 2: Finetuning')
        total_classes = dataset.get_total_seen_classes(task)
        self.mixup(total_classes)
        self.train_stage2(task, model, dataset)

    def get_stage1_loader(self, task, dataset):
        """
        unsupervised loader
        """
        dataloaders = {}
        batch_size = self.args.batch_size
        print(f"Task {task}, loading unlabeled dataset for stage 1...")
        unlabeled_set = dataset.get_unlabeled_set(task)
        unlabeled_loader = get_loader(unlabeled_set, batch_size, self.args)
        dataloaders['cur_unlabeled'] = unlabeled_loader
        return dataloaders

    def compute_iter(self, **kwargs):
        total = self.args.steps * \
            1024 // (self.args.batch_size * self.args.world_size)
        s1_iters = int(total * self.args.train_staeg1_steps_ratio)
        if kwargs['stage'] == 1:
            return s1_iters
        else:
            return total - s1_iters

    def get_stage2_loader(self, task, dataset):
        dataloaders = {}
        batch_sizes = self.compute_batch_size(task)
        print(f"Task {task}, loading labeled dataset...")
        cur_labeled_set = dataset.get_labeled_set(task)
        print(f"Task {task}, labeled set loaded, size {len(cur_labeled_set)}")
        cur_labeled_loader = get_loader(
            cur_labeled_set, batch_sizes[0], self.args)
        if cur_labeled_loader is not None:
            dataloaders['cur_labeled'] = cur_labeled_loader

        print(f"Task {task}, checking buffer...")
        buffer_set = get_and_update_buffer(task, dataset, self.args)
        if buffer_set is not None:
            print(f"Task {task}, buffer set loaded, size {len(buffer_set)}")
            buffer_loader = get_loader(buffer_set, batch_sizes[1], self.args)
            dataloaders['buffer'] = buffer_loader
        else:
            print(f"Task {task}, no buffer")

        return dataloaders

    def train_stage1(self, task, model, dataset):
        start_time = time.time()

        # init optimizer
        blr = compute_lr(self.args) * self.args.train_stage1_lr_scale
        optimizer = self.get_optimizer(model, blr)

        # init dataloader
        train_loader = self.get_stage1_loader(task, dataset)

        # compute iterations
        stage1_iterations = self.compute_iter(stage=1)

        # train
        self.train_iterations(model, optimizer, train_loader,
                              task, stage1_iterations, blr)

        # log overall training time
        total_time = all_reduce_mean(
            time.time() - start_time, self.args.rank, self.args.world_size)
        if self.args.rank == 0 and self.args.wandb.enable:
            wandb_log(f'{task}/total_pretrain_time', total_time)

    def compute_cur_unlabeled_loss(self, model, data):
        """
        Compute the loss of the current unlabeled data
        """
        samples, _ = data
        samples = samples.float().cuda(self.args.rank, non_blocking=True)
        loss = self.args.unlabeled_coef * \
            model(samples, loss_pattern="reconstruction")[0]

        return loss

    def train_stage2(self, task, model, dataset):
        start_time = time.time()

        # init optimizer
        blr = compute_lr(self.args) * self.args.train_stage2_lr_scale
        optimizer = self.get_optimizer(model, blr)

        # init dataloader
        train_loader = self.get_stage2_loader(task, dataset)

        # compute iterations
        stage2_iterations = self.compute_iter(stage=2)

        # train
        self.train_iterations(model, optimizer, train_loader,
                              task, stage2_iterations, blr)

        # log overall training time
        total_time = all_reduce_mean(
            time.time() - start_time, self.args.rank, self.args.world_size)
        if self.args.rank == 0 and self.args.wandb.enable:
            wandb_log(f'{task}/total_finetune_time', total_time)
