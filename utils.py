import os
import datetime
import torch


def get_run_name(args):
    """
    Modify this function to change the format of the run name
    """
    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_"
    run_name += f"{args.dataset}_{args.trainer}"
    return run_name


def save_checkpoint(args, model, val_top1, task):
    if args.rank == 0:
        torch.save(
            {
                'model': model.state_dict(),
                'metric': val_top1,
            },
            f"{args.output_dir}/checkpoint.{task}.pth",
        )
        print(f"Saved the checkpoint for task {task} in {args.output_dir}/checkpoint.{task}.pth")


def find_available_ckpt(dir):
    """
    Find the available checkpoint in the log dir such that 
    the model state is available
    and the metric is available
    """

    ckpt_files = os.listdir(dir)
    ckpt_files = [f for f in ckpt_files if f.startswith("checkpoint.") and f.endswith(".pth")]
    ckpt_files = sorted(ckpt_files)
    available_flag = 0
    while (not available_flag) and len(ckpt_files) > 0:
        # get checkpoint with largest task number
        ckpt_file = ckpt_files.pop()
        ckpt = torch.load(f"{dir}/{ckpt_file}", map_location='cpu')
        if 'model' in ckpt and 'metric' in ckpt:
            available_flag = 1
            task = int(ckpt_file.split(".")[-2])
            return task+1

    return 0


def resume_from_checkpoint(dir, task, model):
    ckpt = torch.load(f"{dir}/checkpoint.{task}.pth", map_location='cpu')
    if 'projection_head.weight' in ckpt['model'].keys():
        classification_head = ckpt['model']['projection_head.weight'].shape[0]
        model.adaptation(classification_head)
    model.load_state_dict(ckpt['model'])
    val_top1 = ckpt['metric']
    print(f"Resumed from checkpoint for task {task}")
    return model, val_top1

def sanity_check_distributed_models(model, ddp_model):
    """
    Check if the model and ddp_model have the same parameters
    """
    for (n1,p1), (n2,p2 )in zip(model.named_parameters(), ddp_model.module.named_parameters()):
        assert n1 == n2, f"Model and DDP model have different parameters: {n1} "
        assert torch.allclose(p1, p2), f"Model and DDP model have different parameters: {n1} "

def wrap_model(model, find_unused_parameters, args):
    if args.distributed:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module        
        model = torch.nn.parallel.DistributedDataParallel(model.to(args.rank), device_ids=[args.rank], find_unused_parameters=find_unused_parameters)
    else:
        model = model.to(args.rank)
    return model
    