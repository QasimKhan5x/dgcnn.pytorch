import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group('nccl')


def prepare_dl(dataset, rank, drop_last, shuffle, world_size, batch_size, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=shuffle, drop_last=drop_last)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                            num_workers=num_workers, drop_last=drop_last, shuffle=shuffle,
                            sampler=sampler)
    return dataloader


def main(rank, world_size):
    # setup the process groups
    setup()
    # prepare the dataloader
    dataloader = prepare_dl(...)

    # instantiate the model(it's your own model) and move it to the right device
    model = Model().to(os.environ)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = Your_Optimizer()
    loss_fn = Your_Loss()
    for epoch in epochs:
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)

        for step, x in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            label = x['label']

            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
    cleanup()


def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    # suppose we have 3 gpus
    world_size = 3
    mp.spawn(
        main,
        args=(world_size),
        nprocs=world_size
    )
