# main file to initialize the ddp model

import os
import sys
import builtins
import hydra
import yaml
import datetime
import importlib
from omegaconf import OmegaConf
from argparse import Namespace

import random
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import trainer as trainer_hub
import dataset as dataset_hub
from utils import *
from metric import ContinaulMetric

# fix random seed
os.environ['TORCH_DISTRIBUTED_DEBUG'] ='DETAIL'

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# use hydra to load the configuration file
@hydra.main(config_path="config", config_name="base", version_base="1.3")
def main(args):
    # convert the args to namespace version, so that we can assign values to it
    OmegaConf.set_struct(args, False)

    # resume the model if needed and possible
    if not os.path.exists(args.resume):
        args.resume = None
    if args.resume and not args.evaluate_only:
        args = yaml.load(open(f"{args.resume}/config.yaml", 'r'), Loader=yaml.FullLoader)
        args = Namespace(**args)
        # check the resume task
        args.starting_task = find_available_ckpt(args.resume)
    else:
        args.starting_task = 0

        # log configuration and save args
        args.run_name = get_run_name(args)
        args.output_dir = os.path.join(args.base_dir, args.run_name)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
            # write the configuration file
            OmegaConf.save(args, f)

    # set the random seed
    if args.dist_url.startswith("tcp://"):  # if dist url is tcp, change the port to a random one before seed everything
        args.dist_url = args.dist_url[:-5] + str(random.randint(10000, 30000))
    seed_everything(args.seed)

    

    # initialize the multiprocessing distributed training
    args.world_size = args.n_gpu_per_node * args.nodes
    if args.world_size == 1:
        args.distributed = False
    if args.distributed:
        print(OmegaConf.to_yaml(args))

        # initialize the process group
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, ),
        )
    else:
        print(OmegaConf.to_yaml(args))
        main_worker(0, args)


def main_worker(gpu, args):

    # setup the current gpu
    args.rank = gpu

    # initialize the distributed training
    if args.distributed:
        setup_print_for_distributed(args.rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print(f"Initialized the process group: {args.rank}")

    # initialize the logger
    if args.wandb.enable and args.rank == 0:
        import wandb
        wandb.init(project=args.wandb.project, name=args.run_name,
                   config=OmegaConf.to_container(args), dir=args.output_dir)

    # initialize the metrics
    val_top1 = ContinaulMetric(args, per_task_evaluation=(not args.light_evaluate))

    # initialize the dataset
    dataset = getattr(dataset_hub, args.dataset)(args)

    # initialize the trainer
    trainer = getattr(trainer_hub, args.trainer)(args)

    # initialize model
    model = trainer.initialize_model()
    print(f"Check DDP model initializing...")
    model = wrap_model(model, args.model.find_unused_parameters, args)


    # resume from checkpoint
    if args.starting_task > 0:
        model, val_top1 = resume_from_checkpoint(args.resume, args.starting_task, model)

    # ————————Start training——————————————————
    for task in range(args.starting_task, args.split):
        print(f"Start training task {task}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module.cpu()

        # adapt model for new classes
        new_classes = dataset.get_new_classes(task)
        if new_classes > 0:
            model = model.module.cpu() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.cpu()
            model.adaptation(new_classes)
            print("Current classification head: ", model.projection_head)
            print(f"Check DDP model re-initializing due to head adaptation...")
            model = wrap_model(model, args.model.find_unused_parameters, args)

        if args.evaluate_only:
            model, _ = resume_from_checkpoint(args.resume, task, model)
            model = wrap_model(model, False, args)
            trainer.evaluate(task, model, dataset, val_top1)
            continue

        # train the model
        trainer.train(task, model, dataset)
        if args.distributed:
            torch.distributed.barrier()
            torch.cuda.synchronize(0)

        if not args.train_only:
            trainer.evaluate(task, model, dataset, val_top1)
            if args.distributed:
                torch.distributed.barrier()
                torch.cuda.synchronize(0)

        # save the checkpoint
        if args.rank == 0:
            save_checkpoint(args, model, val_top1, task)
        
        if args.debug and task > 2:
            break

        # -----------------End training-------------------

    if args.wandb.enable and args.rank == 0:
        wandb.finish()


if __name__ == '__main__':
    main()


def setup_print_for_distributed(rank):
    buildin_print = builtins.print

    def print(*args, **kwargs):
        if rank == 0:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            buildin_print(f'[{now_str}]', *args, **kwargs)
    builtins.print = print
