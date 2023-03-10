import cv2
from numpy.core.fromnumeric import sort
import tqdm
import json
import copy
import Polygon
import numpy as np
from .scitsr.eval import json2Relations, eval_relations


def parse_layout(spans, num_rows, num_cols):
    layout = np.full([num_rows, num_cols], -1, dtype=np.int)
    cell_count = 0
    for x1, y1, x2, y2 in spans:
        layout[y1:y2+1, x1:x2+1] = cell_count
        cell_count += 1

    cells_id = list()
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            cell_id = layout[row_idx, col_idx]
            if cell_id in cells_id:
                layout[row_idx, col_idx] = cells_id.index(cell_id)
            else:
                layout[row_idx, col_idx] = len(cells_id)
                cells_id.append(cell_id)
    return layout


def parse_cells(layout, spans, row_segments, col_segments, lines):
    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout == cell_id)
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        x1 = col_segments[x1]
        x2 = col_segments[x2+1]
        y1 = row_segments[y1]
        y2 = row_segments[y2+1]
        cell = dict(
            segmentation=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        )
        cells.append(cell)

    extend_cell_lines(cells, lines)

    return cells


def extend_cell_lines(cells, lines):
    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    lines = copy.deepcopy(lines)

    cells_poly = [segmentation_to_polygon(item['segmentation']) for item in cells]
    lines_poly = [segmentation_to_polygon(item['segmentation']) for item in lines]

    cells_lines = [[] for _ in range(len(cells))]

    for line_idx, line_poly in enumerate(lines_poly):
        if line_poly.area() == 0:
            continue
        line_area = line_poly.area()
        max_overlap = 0
        max_overlap_idx = None
        for cell_idx, cell_poly in enumerate(cells_poly):
            overlap = (cell_poly & line_poly).area() / line_area
            if overlap > max_overlap:
                max_overlap_idx = cell_idx
                max_overlap = overlap
        if max_overlap > 0:
            cells_lines[max_overlap_idx].append(line_idx)
    lines_y1 = [segmentation_to_bbox(item['segmentation'])[1] for item in lines]
    cells_lines = [sorted(item, key=lambda idx: lines_y1[idx]) for item in cells_lines]

    for cell, cell_lines in zip(cells, cells_lines):
        transcript = []
        for idx in cell_lines:
            transcript.extend(lines[idx]['transcript'])
        cell['transcript'] = transcript


def segmentation_to_bbox(segmentation):
    x1 = min([min([pt[0] for pt in contour]) for contour in segmentation])
    y1 = min([min([pt[1] for pt in contour]) for contour in segmentation])
    x2 = max([max([pt[0] for pt in contour]) for contour in segmentation])
    y2 = max([max([pt[1] for pt in contour]) for contour in segmentation])
    return [x1, y1, x2, y2]


def cal_cell_spans(table):
    layout = table['layout']
    num_cells = len(table['cells'])
    cells_span = list()
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout == cell_id)
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return cells_span


def pred_result_to_table(table, pred_result):
    # gt ocr result
    lines = [dict(segmentation=cell['segmentation'], transcript=cell['transcript']) for cell in table['cells'] if 'bbox' in cell.keys()]

    row_segments, col_segments, divide, spans = pred_result
    num_rows = len(row_segments) - 1
    num_cols = len(col_segments) - 1

    layout = parse_layout(spans, num_rows, num_cols)
    cells = parse_cells(layout, spans, row_segments, col_segments, lines)
    head_rows = list(range(0, divide))
    body_rows = list(range(divide, num_rows))
    
    table = dict(
        layout=layout,
        head_rows=head_rows,
        body_rows=body_rows,
        cells=cells
    )
    
    return table


def table_to_relations(table):
    cell_spans = cal_cell_spans(table)
    contents = [''.join(cell['transcript']).split() for cell in table['cells']]
    relations = []
    for span, content in zip(cell_spans, contents):
        x1, y1, x2, y2 = span
        relations.append(dict(start_row=y1, end_row=y2, start_col=x1, end_col=x2, content=content))
    return dict(cells=relations)


def cal_f1(label, pred):
    label = json2Relations(label, splitted_content=True)
    pred = json2Relations(pred, splitted_content=True)
    precision, recall = eval_relations(gt=[label], res=[pred], cmp_blank=True)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return [precision, recall, f1]


def single_process(labels, preds):
    scores = dict()
    for key in tqdm.tqdm(labels.keys()):
        pred = preds.get(key, '')
        label = labels.get(key, '')
        score = cal_f1(label, pred)
        scores[key] = score
    return scores


def _worker(labels, preds,  keys, result_queue):
    for key in keys:
        label = labels.get(key, '')
        pred = preds.get(key, '')
        score = cal_f1(label, pred)
        result_queue.put((key, score))


def multi_process(labels, preds, num_workers):
    import multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    keys = list(labels.keys())
    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker,
            args=(
                labels,
                preds,
                keys[worker_idx::num_workers],
                result_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    scores = dict()
    tq = tqdm.tqdm(total=len(keys))
    for _ in range(len(keys)):
        key, val = result_queue.get()
        scores[key] = val
        P, R, F1 = (100 * np.array(list(scores.values()))).mean(0).tolist()
        tq.set_description('P: %.2f, R: %.2f, F1: %.2f' % (P, R, F1), False)
        tq.update()
    
    return scores


def evaluate_f1(labels, preds, num_workers=0):
    preds = {idx: pred for idx, pred in enumerate(preds)}
    labels = {idx: label for idx, label in enumerate(labels)}
    if num_workers == 0:
        scores = single_process(labels, preds)
    else:
        scores = multi_process(labels, preds, num_workers)
    sorted_idx = sorted(list(range(len(list(scores)))), key=lambda idx: list(scores.keys())[idx])
    scores = [scores[idx] for idx in sorted_idx]
    return scores
