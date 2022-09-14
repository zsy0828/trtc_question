import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch_geometric.datasets import Reddit
import scipy.sparse as sp
import numpy as np
import torch.multiprocessing as mp

from gglspeedup.gpusample import GPUSampler

from tt import ttt_havetrtc, ttt_havenotrtc, ttt_haverndrtc


def run(rank, world_size, tt):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.torch.cuda.set_device(rank)

    print(tt)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    world_size = 2

    # tt = ttt_havetrtc(np.arange(10))
    # mp.spawn(run, args=(world_size, tt), nprocs=world_size, join=True)
    tt = ttt_haverndrtc(np.arange(10))
    mp.spawn(run, args=(world_size, tt), nprocs=world_size, join=True)
    # tt = ttt_havenotrtc(np.arange(10))
    # mp.spawn(run, args=(world_size, tt), nprocs=world_size, join=True)
