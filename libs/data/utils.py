import math
from libs.utils.format_translate import table_to_html, format_html
import numpy as np


class InvalidFormat(Exception):
    pass


def segmentation_to_bbox(segmentation):
    x1 = min([pt[0] for contour in segmentation for pt in contour])
    y1 = min([pt[1] for contour in segmentation for pt in contour])
    x2 = max([pt[0] for contour in segmentation for pt in contour])
    y2 = max([pt[1] for contour in segmentation for pt in contour])
    return (x1, y1, x2, y2)


def cal_cell_bbox(table):
    cells_bbox = list()
    for cell in table['cells']:
        if 'segmentation' not in cell:
            cell_bbox = None
        else:
            segmentation = list()
            if 'sublines' in cell:
                for subline in cell['sublines']:
                    segmentation.extend(subline['segmentation'])
            if len(segmentation) == 0:
                segmentation = cell['segmentation']
            if len(segmentation) == 0:
                cell_bbox = None
            else:
                cell_bbox = segmentation_to_bbox(segmentation)
            cells_bbox.append(cell_bbox)
    return cells_bbox


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

def cal_fg_bg_span(spans, edge):
    num_span = len(spans)
    bg_spans = list()
    for idx in range(num_span):
        if spans[idx] is None:
            continue
        if idx == 0:
            if spans[idx][0] <= 0:
                continue
        else:
            if spans[idx-1] is None:
                continue
            if spans[idx][0] <= spans[idx-1][1]:
                continue
        if idx == num_span - 1:
            if spans[idx][1] >= edge:
                continue
        else:
            if spans[idx+1] is None:
                continue
            if spans[idx][1] >= spans[idx+1][0]:
                continue

        bg_spans.append(spans[idx])

    fg_spans = list()
    for idx in range(num_span+1):
        if idx == 0:
            s = 0
        else:
            if spans[idx-1] is None:
                continue
            s = spans[idx-1][1]
        
        if idx == num_span:
            e = edge
        else:
            if spans[idx] is None:
                continue
            e = spans[idx][0]
        
        if e <= s:
            continue
        
        fg_spans.append([s, e])

        return fg_spans, bg_spans


def shrink_spans(spans, size):
    new_spans = list()
    for idx, (start, end) in enumerate(spans):
        if idx == 0:
            if start <= 0:
                start = 1
        else:
            _, pre_end = spans[idx - 1]
            if start <= pre_end:
                shrink_distance = pre_end - start + 1
                start = start + math.ceil(shrink_distance / 2)
        
        if idx == len(spans) - 1:
            if end >= size:
                end = size - 1
        else:
            next_start, _ = spans[idx + 1]
            if end >= next_start:
                shrink_distance = end - next_start + 1
                end = end - math.ceil(shrink_distance / 2)
        if end - start < 1:
            raise InvalidFormat()

        new_spans.append([start, end])
    return new_spans


def cal_row_span(table, cells_span, cells_bbox, height):
    layout = table['layout']
    rows_span = list()
    for row_idx in range(layout.shape[0]):
        row = layout[row_idx, :]
        y1s = list()
        y2s = list()
        for cell_id in row:
            cell_span = cells_span[cell_id]
            cell_bbox = cells_bbox[cell_id]
            if (cell_span[1] == row_idx) and (cell_bbox is not None):
                y1s.append(cell_bbox[1])
            if (cell_span[3] == row_idx) and (cell_bbox is not None):
                y2s.append(cell_bbox[3])

        if (len(y1s) > 0) and (len(y2s) > 0):
            y1 = min(max(1, min(y1s)), height - 1)
            y2 = min(max(1,max(y2s) + 1), height - 1)
            rows_span.append([y1, y2])
        else:
            raise InvalidFormat()
    rows_span = shrink_spans(rows_span, height)
    rows_fg_span, rows_bg_span = cal_fg_bg_span(rows_span, height)
    return rows_fg_span, rows_bg_span


def cal_col_span(table, cells_span, cells_bbox, width):
    layout = table['layout']
    cols_span = list()
    for col_idx in range(layout.shape[1]):
        col = layout[:, col_idx]
        x1s = list()
        x2s = list()
        for cell_id in col:
            cell_span = cells_span[cell_id]
            cell_bbox = cells_bbox[cell_id]
            if (cell_span[0] == col_idx) and (cell_bbox is not None):
                x1s.append(cell_bbox[0])
            if (cell_span[2] == col_idx) and (cell_bbox is not None):
                x2s.append(cell_bbox[2])
        
        if (len(x1s) > 0) and (len(x2s) > 0):
            x1 = min(max(1, min(x1s)), width - 1)
            x2 = min(max(1, max(x2s) + 1), width - 1)
            cols_span.append([x1, x2])
        else:
            raise InvalidFormat()
    cols_span = shrink_spans(cols_span, width)
    cols_fg_span, cols_bg_span = cal_fg_bg_span(cols_span, width)
    return cols_fg_span, cols_bg_span


def extract_fg_bg_spans(table, image_size):
    width, height = image_size
    cells_bbox = cal_cell_bbox(table)
    cells_span = cal_cell_spans(table)
    # cal rows fg bg span
    rows_fg_span, rows_bg_span = cal_row_span(table, cells_span, cells_bbox, height)
    #cal cols fg bg span
    cols_fg_span, cols_bg_span = cal_col_span(table, cells_span, cells_bbox, width)
    return rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span
