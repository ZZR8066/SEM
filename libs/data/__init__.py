import torch
from torch.utils.data.distributed import DistributedSampler
from .batch_sampler import BucketSampler
from .dataset import LRCRecordLoader
from .dataset import Dataset, collate_func
from libs.utils.comm import distributed, get_rank, get_world_size
from . import transform as T


def create_train_dataloader(vocab, lrcs_path, num_workers, max_batch_size, max_pixel_nums, bucket_seps, data_root_dir):
    loaders = list()
    for lrc_path in lrcs_path:
        loader = LRCRecordLoader(lrc_path, data_root_dir)
        loaders.append(loader)

    transforms = T.Compose([
        T.TableToLabel(vocab),
        T.CalRowColSpans(),
        T.CalCellSpans(),
        T.CalHeadBodyDivide(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Dataset(loaders, transforms)
    batch_sampler = BucketSampler(dataset, get_world_size(), get_rank(), max_pixel_nums=max_pixel_nums, max_batch_size=max_batch_size,seps=bucket_seps)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_func,
        batch_sampler=batch_sampler
    )
    return dataloader


def create_valid_dataloader(vocab, lrc_path, num_workers, batch_size, data_root_dir):
    loader = LRCRecordLoader(lrc_path, data_root_dir)

    transforms = T.Compose([
        T.TableToLabel(vocab),
        T.CalRowColSpans(),
        T.CalCellSpans(),
        T.CalHeadBodyDivide(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Dataset([loader], transforms)
    if distributed():
        sampler = DistributedSampler(dataset, get_world_size(), get_rank(), True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_func,
            sampler=sampler,
            drop_last=False
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_func,
            shuffle=False,
            drop_last=False
        )
    return dataloader
