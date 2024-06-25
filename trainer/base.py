import sys
import time
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from timm.data.mixup import Mixup
import torch.nn.functional as F

import model as model_hub
from data import get_loader, get_and_update_buffer
from trainer.utils import NativeScalerWithGradNormCount as NativeScaler
from trainer.utils import *
from trainer.lr_decay import param_groups_lrd
from trainer.lr_sched import adjust_learning_rate
from metric import AverageMeter


class SoftTargetCrossEntropy(nn.Module):
    """
    Soft target cross-entropy loss with mask
    Adopted from timm
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # mask out the logits of specific classes
        loss = -target * F.log_softmax(x, dim=-1)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.repeat(loss.size(0), 1)
            loss = loss * mask
            loss = loss.sum(dim=-1)
        else:
            loss = loss.sum(dim=-1)
        return loss.mean()


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args

    def initialize_model(self):

        model = getattr(model_hub, self.args.model.name)()
        print (model)
        if self.args.model.pre_trained is not None:
            model.load_state_dict(torch.load(self.args.model.pre_trained, map_location='cpu')['model'], strict=False)
            print(f"Loaded the pre-trained model from {self.args.model.pre_trained}")
        
        return model

    def mixup(self, total_classes):
        mixup_active = self.args.augment.mixup > 0 or self.args.augment.cutmix > 0 or self.args.augment.cutmix_minmax is not None
        if mixup_active:
            self.mixup_fcn = Mixup(
                mixup_alpha=self.args.augment.mixup,
                cutmix_alpha=self.args.augment.cutmix,
                cutmix_minmax=self.args.augment.cutmix_minmax,
                prob=self.args.augment.mixup_prob,
                switch_prob=self.args.augment.mixup_switch_prob,
                mode=self.args.augment.mixup_mode,
                label_smoothing=self.args.augment.label_smoothing,
                num_classes=total_classes
            )
        print(f"Mixup is active: {total_classes} classes")

    def get_optimizer(self, model, lr):

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            param_model = model.module
        else:
            param_model = model

        param_groups = param_groups_lrd(
            param_model, self.args, weight_decay=self.args.weight_decay,
            no_weight_decay_list=param_model.no_weight_decay(),
            layer_decay=self.args.layer_decay)

        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        return optimizer

    def compute_batch_size(self, task):
        """
        return the batch size for the current task, buffer
        """

        assert (self.args.sampling in ['batchmix', 'uniform']) or self.args.replay_buffer_size == 0
        if self.args.replay_buffer_size == 0 or task == 0:
            batch_sizes = [self.args.batch_size, 0]
        elif self.args.sampling == 'uniform':
            batch_sizes = [0, self.args.batch_size]
        else:
            batch_sizes = [self.args.batch_size // 2, self.args.batch_size // 2]
        print(f"Task {task}, labeled batch size {batch_sizes[0]}, buffer batch size {batch_sizes[1]}")
        return batch_sizes

    def get_dataloaders(self, task, dataset):
        dataloaders = {}
        batch_sizes = self.compute_batch_size(task)
        print(f"Task {task}, loading labeled dataset...")
        cur_labeled_set = dataset.get_labeled_set(task)
        print(f"Task {task}, labeled set loaded, size {len(cur_labeled_set)}")
        cur_labeled_loader = get_loader(cur_labeled_set, batch_sizes[0], self.args)
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

    def supervised_entropy_loss(self, model, data, mask_logits=False):

        samples, targets = data

        unique_classes = targets.unique().cuda(self.args.rank, non_blocking=True)
        samples, targets = samples.float().cuda(self.args.rank, non_blocking=True), targets.cuda(self.args.rank, non_blocking=True)
        if self.mixup_fcn is not None:
            if len(samples) % 2 != 0:
                samples = samples[:-1]
                targets = targets[:-1]
            samples, targets = self.mixup_fcn(samples, targets)
            criterion = SoftTargetCrossEntropy()
        else:
            criterion = CrossEntropyLoss()

        with torch.cuda.amp.autocast():

            logits = model(samples, loss_pattern='classification')

        if mask_logits:
            if self.args.cl_setting == 'class_incremental':
                loss = criterion(logits[:, -self.args.new_classes:], targets[:, -self.args.new_classes:])
            else:

                mask = torch.zeros(logits.size(1), dtype=torch.bool).to(logits.device)
                mask.scatter_(0, torch.tensor(unique_classes), True)
                if len(targets.size()) == 1:
                    logtis.scatter_(1, mask, -float('inf'))
                    loss = criterion(logits, targets)
                else:
                    loss = criterion(logits, targets, mask)
        else:
            loss = criterion(logits, targets)

        return loss

    def compute_cur_labeled_loss(self, model, data):
        return self.supervised_entropy_loss(model, data, mask_logits=self.args.mask_logits)

    def compute_buffer_loss(self, model, data):
        return self.supervised_entropy_loss(model, data)
    
    def compute_iter(self, **kwargs):
        """
        Compute the number of iterations 
        """
        return self.args.steps * 1024 // (self.args.batch_size * self.args.world_size)

    def train(self, task, model, dataset):
        # init mixup
        total_classes = dataset.get_total_seen_classes(task)
        self.mixup(total_classes)

        # get optimizer
        blr = compute_lr(self.args)
        optimizer = self.get_optimizer(model, blr)

        # get dataloader
        dataloaders = self.get_dataloaders(task, dataset)

        # compute total iterations
        total_iter = self.compute_iter()

        start_time = time.time()
        
        # +++++++++++++++ Start training ++++++++++ß+++++
        print(f"Start training task {task}")
        self.train_iterations(model, optimizer, dataloaders, task, total_iter,  blr)
        # +++++++++++++++ End training +++++++++++++++

        # log overall training time
        total_time = all_reduce_mean(time.time() - start_time, self.args.rank, self.args.world_size)
        if self.args.rank == 0 and self.args.wandb.enable:
            wandb_log(f'{task}/total_train_time', total_time)

    def train_iterations(self, model, optimizer, dataloaders, task, total_iter, blr):
        # setup logging
        epoch_counts = {k: -1 for k in dataloaders.keys()}

        iter_per_epochs = {k: len(v) for k, v in dataloaders.items()}
        loss_scaler = NativeScaler()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        model.train(True)
        optimizer.zero_grad()

        
        accumulation = compute_accumulation_steps(self.args)

        iterators = {k: None for k, v in dataloaders.items()}

        
        for iteration in range(total_iter):
            step = iteration//accumulation
            end = time.time()
            # check if we need gradient step at the current step
            do_step = (iteration+1) % accumulation == 0 or iteration+1 == total_iter

            if (iteration) % accumulation == 0:
                lr = adjust_learning_rate(optimizer, iteration, total_iter, blr, self.args.min_lr, self.args.warmup)

            # iterate the dataloaders if we need
            for k, v in dataloaders.items():
                if iteration % iter_per_epochs[k] == 0:
                    epoch_counts[k] += 1
                    if self.args.distributed:
                        v.sampler.set_epoch(epoch_counts[k])
                    iterators[k] = iter(v)

            # get the data
            datas = {k: next(v) for k, v in iterators.items()}
            cur_data_time = all_reduce_mean(time.time() - end, self.args.rank, self.args.world_size)
            data_time.update(cur_data_time)

            # compute loss
            all_loss = {}
            for k, data in datas.items():
                all_loss[k] = getattr(self, f'compute_{k}_loss')(model, data)
            del datas

            torch.cuda.empty_cache()

            loss = sum(all_loss.values()) 

            if not math.isfinite(loss):
                print(f"Loss is {loss}, stopping training")
                break

            # backprop
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=do_step)

            if do_step:
                optimizer.zero_grad()

            # measure elapsed time
            cur_batch_time = all_reduce_mean(time.time() - end, self.args.rank, self.args.world_size)
            batch_time.update(cur_batch_time)

            # logging
            for k in all_loss.keys():
                all_loss[k] = all_reduce_mean(all_loss[k], self.args.rank, self.args.world_size)
            total_loss = all_reduce_mean(loss, self.args.rank, self.args.world_size)
            if self.args.rank == 0 and self.args.wandb.enable and do_step:
                wandb_log(f'{task}/lr', lr, step)
                wandb_log(f'{task}/total_loss', total_loss, step)
                for k in all_loss.keys():
                    wandb_log(f'{task}/{k}_loss',  all_loss[k], step)
                wandb_log(f'{task}/batch_time', batch_time.avg, step)
                wandb_log(f'{task}/data_time', data_time.avg, step)

            # print
            if iteration % self.args.print_freq == 0:
                printing_content = f"Task [{task}/{self.args.split}], Iteration [{iteration}/{total_iter}], Step [{step}/{self.args.steps}]"
                for k, v in epoch_counts.items():
                    printing_content += f", {k} epoch [{v}], loss {all_loss[k]:.4f}"
                print(printing_content)
                sys.stdout.flush()

    def evaluate(self, task, model, dataset, metric):

        print(f"Start evaluating task {task}")

        total_eval_task = 1 if metric.per_task_evaluation else task+1
        criterion = CrossEntropyLoss()

        batch_time = AverageMeter()

        model.eval()
        with torch.no_grad():
            for eval_task in range(total_eval_task):
                loss = AverageMeter()
                top1 = AverageMeter()

                eval_set = dataset.get_eval_set(eval_task, per_task_eval=metric.per_task_evaluation)
                eval_loader = get_loader(eval_set, self.args.batch_size, self.args, is_train=False)

                for i, (samples, target) in enumerate(eval_loader):
                    end = time.time()
                    samples, target = samples.float().cuda(self.args.rank, non_blocking=True), target.cuda(self.args.rank, non_blocking=True)
                    logits = model(samples, loss_pattern='classification')

                    loss_val = criterion(logits, target)
                    prec1 = accuracy(logits, target, topk=(1,))[0]
                    prec1 = all_reduce_mean(prec1, self.args.rank, self.args.world_size)

                    loss.update(all_reduce_mean(loss_val, self.args.rank, self.args.world_size), samples.size(0))
                    top1.update(prec1, samples.size(0))
                    metric.update(eval_task, task, prec1, samples.size(0))
                    batch_time.update(all_reduce_mean(time.time() - end, self.args.rank, self.args.world_size))
        metric.summarize(task)

        print(f"Task {task}, task average accuracy {metric.task_average}")
        print(f"Task {task}, task learning average accuracy {metric.task_learning_average}")
        print(f"Task {task}, backward transfer {metric.backward_transfer}")

        if self.args.rank == 0 and self.args.wandb.enable:
            wandb.log({f"task average accuracy": metric.task_average})
            wandb.log({f"task learning average accuracy": metric.task_learning_average})
            wandb.log({f"backward transfer": metric.backward_transfer})
