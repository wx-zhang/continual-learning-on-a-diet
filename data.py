

import torch
from torch.distributed import broadcast
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, ConcatDataset, Subset






def get_loader( train_set, batch_size, args, is_train=True):
        if train_set is None or batch_size == 0 :
             return None
        
        if args.distributed:
            train_sampler = DistributedSampler(train_set)
        else:
            train_sampler = None
        if is_train:
            train_loader = DataLoader( train_set, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)
        else:
             train_loader = DataLoader( train_set, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=False)
        return train_loader



def get_and_update_buffer(task, dataset, args):
    if args.replay_buffer_size == 0:
        return None
    else:
        if args.replay_before:
            mem = update_buffer(args, task, dataset.get_labeled_set(task))
            memory =[]
            for i in range(len(mem)):
                memory.append(Subset(dataset.get_labeled_set(i), mem[i]))
            memory = ConcatDataset(memory)
        else:
            if task > 0:
                mem = torch.load(f"{args.output_dir}/buffer.torchSave") 
                memory =[]
                for i in range(len(mem)):
                    memory.append(Subset(dataset.get_labeled_set(i), mem[i]))
                memory = ConcatDataset(memory) 
            else:
                memory = None
            _ = update_buffer(args, task, dataset.get_labeled_set(task))
        return memory

def get_buffer( dataset, args):
    mem = torch.load(f"{args.output_dir}/buffer.torchSave") 
    memory =[]
    for i in range(len(mem)):
        memory.append(Subset(dataset.get_labeled_set(i), mem[i]))
    memory = ConcatDataset(memory) 
    return memory




def update_buffer(args, task, cur_set):
    """
    Task balanced buffer update
    """
    if task > 0:
        last_memory = torch.load(f"{args.output_dir}/buffer.torchSave").to(args.rank)
        prev_len = last_memory.size(1)
    else:
        prev_len=-1
    cur_len = len(cur_set)


    mem_len = min([item for item in [args.replay_buffer_size//(task+1), cur_len, prev_len] if item > 0])
    
    cur_index = torch.arange(0, mem_len).cuda(args.rank)

    if args.rank == 0:
        cur_index = torch.randperm(cur_len)[:mem_len].sort()[0].cuda(args.rank)

    if args.distributed:
        handle = broadcast(cur_index, src=0, async_op=True)
        handle.wait()
        torch.distributed.barrier()
        torch.cuda.synchronize(0)

    if task>0:
            
        mem = last_memory[:, :mem_len]
        mem = torch.cat([mem, cur_index.unsqueeze(0)], dim=0)
    else:
        mem = cur_index.unsqueeze(0)

    print (f"Task {task}, buffer {mem}")
    
    if args.rank == 0:
        torch.save(mem, f"{args.output_dir}/buffer.torchSave")
    return mem



        