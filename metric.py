import numpy as np

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum, self.val], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        self.val = t[2]
        self.avg = self.sum / self.count


class ContinaulMetric(object):
    def __init__(self,args, per_task_evaluation=False):


        self.total_task = args.split 
        self.per_task_evaluation = per_task_evaluation

        
        self.reset()

    def reset(self):
        self.task_average = 0
        self.sample_average = 0
        self.backward_transfer = 0
        self.task_learning_average = 0
        
        if not self.per_task_evaluation:
            self.task_matrix = np.zeros((self.total_task,1))
        else:

            self.task_matrix = np.zeros((self.total_task,self.total_task))

        self.task_metrics = {}
        for task in range(self.task_matrix.shape[0]):
            self.task_metrics[task] = {}
            for data_task in range(self.task_matrix.shape[1]):
                self.task_metrics[task][data_task] = AverageMeter()

            
        
    def update(self, data_task, model_task, val, n=1):
        if not self.per_task_evaluation:
            data_task = 0
        
        self.task_metrics[model_task][data_task].update(val, n)
        
    def summarize(self, task):
        for model_task in range(self.task_matrix.shape[0]):
            for data_task in range(self.task_matrix.shape[1]):
                self.task_matrix[model_task][data_task] = self.task_metrics[model_task][data_task].avg
        if not self.per_task_evaluation:
            self.task_average = self.task_matrix[task][0]
        else:              
            task_accuracy = [np.mean(self.task_matrix[t][:t+1]) for t in range(self.task_matrix.shape[0])]
            self.task_average = np.mean(task_accuracy[:task+1])
            self.task_learning_average = np.diag(self.task_matrix)[:task+1].mean()
            if task > 0:
                self.backward_transfer = np.mean(np.max(self.task_matrix[:task]-self.task_matrix[task],axis=1)[:task])
        
        
            
        
        
        
        