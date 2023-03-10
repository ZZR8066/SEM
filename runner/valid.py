import sys
import json
sys.path.append('./')
sys.path.append('../')
import os
import tqdm
import torch
import numpy as np
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1
from libs.utils.checkpoint import load_checkpoint
from libs.utils.comm import synchronize, all_gather


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrc", type=str, default=None)
    parser.add_argument("--cfg", type=str, default='default')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    setup_config(args.cfg)
    if args.lrc is not None:
        cfg.valid_lrc_path = args.lrc

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('Line Detect Model', cfg.work_dir, 'valid.log')
    logger.info('Use config: %s' % args.cfg)
    logger.info('Evaluate Dataset: %s' % cfg.valid_lrc_path)


def valid(cfg, dataloader, model):
    model.eval()
    total_label_relations = list()
    total_pred_relations = list()
    total_relations_metric = list()
    
    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        ids = data_batch['ids']
        images_size = data_batch['images_size']
        images = data_batch['images'].to(cfg.device)
        tables = data_batch['tables']

        pred_result, _ = model(images, images_size)

        # pred
        pred_tables = [
            pred_result_to_table(tables[batch_idx],
                (pred_result[0][batch_idx], pred_result[1][batch_idx], \
                    pred_result[2][batch_idx], pred_result[3][batch_idx])
            ) \
            for batch_idx in range(len(ids))
        ]
        pred_relations = [table_to_relations(table) for table in pred_tables]
        total_pred_relations.extend(pred_relations)
        # label
        label_relations = []
        for table in tables:
            with open(table['label_path'], 'r') as f:
                label_relations.append(json.load(f))
        total_label_relations.extend(label_relations)

    # cal P, R, F1
    total_relations_metric = evaluate_f1(total_label_relations, total_pred_relations, num_workers=40)
    P, R, F1 = np.array(total_relations_metric).mean(0).tolist()
    F1 = 2 * P * R / (P + R)
    logger.info('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))

    return (F1, )


def main():
    init()

    valid_dataloader = create_valid_dataloader(
        cfg.vocab,
        cfg.valid_lrc_path,
        cfg.valid_num_workers,
        cfg.valid_batch_size
    )
    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(valid_dataloader.dataset),
                len(valid_dataloader.batch_sampler),
                valid_dataloader.batch_size
            )
    )

    model = build_model(cfg)
    model.cuda()
    
    load_checkpoint(cfg.eval_checkpoint, model)
    logger.info('Load checkpoint from: %s' % cfg.eval_checkpoint)

    with torch.no_grad():
        valid(cfg, valid_dataloader, model)


if __name__ == '__main__':
    main()
