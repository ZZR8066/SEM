import os
import glob
import tqdm
import numpy as np
from utlis.list_record_cache import ListRecordCacher, merge_record_file
from utlis.utlis import get_paths, get_sub_paths, crop_pdf, extract_ocr, refine_table, visualize_table


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str, default=None)
    parser.add_argument('dst_dir', type=str, default=None)
    parser.add_argument('-n', '--num_workers', type=int, default=0)
    args = parser.parse_args()
    return args


def single_process(paths, dst_dir):

    output_pdf_dir = os.path.join(dst_dir, 'pdf')
    if not os.path.exists(output_pdf_dir):
        os.makedirs(output_pdf_dir)
    output_img_dir = os.path.join(dst_dir, 'img')
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)    
    output_error_dir = os.path.join(dst_dir, 'error')
    if not os.path.exists(output_error_dir):
        os.makedirs(output_error_dir)    
    output_visual_dir = os.path.join(dst_dir, 'visual')
    if not os.path.exists(output_visual_dir):
        os.makedirs(output_visual_dir)    

    cacher = ListRecordCacher(os.path.join(dst_dir, 'table.lrc'))

    error_paths = []
    error_count = 0
    correct_count = 0
    for id, path in enumerate(tqdm.tqdm(paths)):
        try:
            pdf_path, chunk_path, structure_path = path
            name = os.path.splitext(os.path.basename(pdf_path))[0]

            positions, transcripts = crop_pdf(path, output_pdf_dir)
            table = extract_ocr(path, positions, transcripts)
            table = refine_table(table, os.path.join(output_pdf_dir, name + '.png'), output_img_dir)

            assert os.path.exists(structure_path), print('structure_path is not existed')
            table['label_path'] = structure_path

            visualize_table(os.path.join(output_img_dir, name + '.png'), output_visual_dir, table)

            cacher.add_record(table)
            correct_count += 1
        except:
            error_count += 1
            error_paths.append(path)
            crop_pdf(path, output_error_dir)

    print("correct num: %d, error num: %d " % (correct_count, error_count))
    if len(error_paths) > 0:
        np.save(os.path.join(dst_dir, 'error_paths.npy'), error_paths)
    cacher.close()


def _worker(worker_idx, num_workers, paths, dst_dir, result_queue):

    output_pdf_dir = os.path.join(dst_dir, 'pdf')
    output_img_dir = os.path.join(dst_dir, 'img')
    output_error_dir = os.path.join(dst_dir, 'error')
    output_visual_dir = os.path.join(dst_dir, 'visual')

    cacher = ListRecordCacher(os.path.join(dst_dir, 'table_%d.lrc' % worker_idx))

    error_paths = []
    error_count = 0
    correct_count = 0
    for id, path in enumerate(tqdm.tqdm(paths)):
        try:
            pdf_path, chunk_path, structure_path = path
            name = os.path.splitext(os.path.basename(pdf_path))[0]

            positions, transcripts = crop_pdf(path, output_pdf_dir)
            table = extract_ocr(path, positions, transcripts)
            table = refine_table(table, os.path.join(output_pdf_dir, name + '.png'), output_img_dir)

            assert os.path.exists(structure_path), print('structure_path is not existed')
            table['label_path'] = structure_path

            visualize_table(os.path.join(output_img_dir, name + '.png'), output_visual_dir, table)
            cacher.add_record(table)
            correct_count += 1
        except:
            error_count += 1
            error_paths.append(path)
            crop_pdf(path, output_error_dir)

    result_queue.put((correct_count, error_count, error_paths))


def multi_process(path, dst_dir, num_workers):
    import multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker,
            args=(
                worker_idx,
                num_workers,
                path[worker_idx::num_workers],
                dst_dir,
                result_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    total_correct_count = 0
    total_error_count = 0
    total_error_paths = []
    for _ in range(num_workers):
        correct_count, error_count, error_paths = result_queue.get()
        total_correct_count += correct_count
        total_error_count += error_count
        total_error_paths.extend(error_paths)
    
    print("correct num: %d, error num: %d " % (total_correct_count, total_error_count))
    if len(total_error_paths) > 0:
        np.save(os.path.join(dst_dir, 'error_paths.npy'), total_error_paths)

    # merge each worker lrc
    cache_paths = glob.glob(os.path.join(dst_dir, '*.lrc'))
    merge_record_file(cache_paths, os.path.join(dst_dir, 'table.lrc'))
    for cache_path in cache_paths:
        os.remove(cache_path)


def main():
    args = parse_args()

    paths = get_sub_paths(args.src_dir, ["pdf", "chunk", "structure"], ['.pdf', '.chunk', '.json'])
    # paths = get_paths(args.src_dir, ["pdf", "chunk", "structure"], '/yrfs1/intern/zrzhang6/TSR/Dataset/SciTSR/SciTSR-COMP.list', ['.pdf', '.chunk', '.json'])

    if args.num_workers == 0:
        single_process(paths, args.dst_dir)
    else:
        multi_process(paths, args.dst_dir, args.num_workers)


if __name__ == "__main__":
    main()