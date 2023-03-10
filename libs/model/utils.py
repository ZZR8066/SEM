import cv2
import copy
import torch
import numpy as np
from torch.nn import functional as F

def proposal_colspan(layout, layout_score, srow, scol):

    y, x = torch.where(layout == 1)
    if torch.all(layout[y.min():y.max() + 1, x.min():x.max()+1] == 1):
        return layout, layout_score[y.min():y.max() + 1, x.min():x.max()+1].mean()
    else:
        lf_row = srow
        lf_col = scol
        
        col_count = 0
        for col_ in range(lf_col, x.max() + 1):
            if layout[lf_row, col_] == 1:
                col_count = col_count + 1
            else:
                break
        row_count = 0
        for row_ in range(lf_row, y.max() + 1):
            if torch.all(layout[row_, lf_col: lf_col + col_count] == 1):
                row_count = row_count + 1
            else:
                break
        
        layout[:, :] = 0
        layout[lf_row:lf_row + row_count, lf_col : lf_col + col_count] = 1
        return layout, layout_score[lf_row:lf_row + row_count, lf_col : lf_col + col_count].mean()

def proposal_rowspan(layout, layout_score, srow, scol):

    
    y, x = torch.where(layout == 1)
    if torch.all(layout[y.min():y.max() + 1, x.min():x.max()+1] == 1):
        return layout, layout_score[y.min():y.max() + 1, x.min():x.max()+1].mean()
    else:
        lf_row = srow
        lf_col = scol
        
        row_count = 0
        for row_ in range(lf_row, y.max() + 1):
            if layout[row_, lf_col] == 1:
                row_count = row_count + 1
            else:
                break
        col_count = 0
        for col_ in range(lf_col, x.max() + 1):
            if torch.all(layout[lf_row : lf_row + row_count, col_] == 1):
                col_count = col_count + 1
            else:
                break
        
        layout[:, :] = 0
        layout[lf_row:lf_row + row_count, lf_col : lf_col + col_count] = 1
        return layout, layout_score[lf_row:lf_row + row_count, lf_col : lf_col + col_count].mean()

def proposal_maxcontain(layout, layout_score, srow, scol):

    
    y, x = torch.where(layout == 1)
    if torch.all(layout[y.min():y.max() + 1, x.min():x.max()+1] == 1):
        return layout, layout_score[y.min():y.max() + 1, x.min():x.max()+1].mean()
    else:
        lf_row = srow
        lf_col = scol
        
        layout[:, :] = 0
        layout[lf_row: y.max()+1, lf_col : x.max() + 1] = 1
        return layout, layout_score[lf_row: y.max()+1, lf_col : x.max() + 1].mean()

def proposal_maxrowspan(layout, layout_score, srow, scol):

    
    y, x = torch.where(layout == 1)
    if torch.all(layout[y.min():y.max() + 1, x.min():x.max()+1] == 1):
        return layout, layout_score[y.min():y.max() + 1, x.min():x.max()+1].mean()
    else:
        lf_row = srow
        lf_col = scol
        
        row_count = 1
        for row_ in range(lf_row + 1, y.max() + 1):
            if torch.all(layout[lf_row] == layout[row_]):
                row_count = row_count + 1
            else:
                break
        
        layout[:, :] = 0
        layout[lf_row : lf_row + row_count, lf_col : x.max() + 1] = 1
        return layout, layout_score[lf_row : lf_row + row_count, lf_col : x.max() + 1].mean()
    
def proposal_maxcolspan(layout, layout_score, srow, scol):

    
    y, x = torch.where(layout == 1)
    if torch.all(layout[y.min():y.max() + 1, x.min():x.max()+1] == 1):
        return layout, layout_score[y.min():y.max() + 1, x.min():x.max()+1].mean()
    else:
        lf_row = srow
        lf_col = scol
        
        col_count = 1
        for col_ in range(lf_col + 1, x.max() + 1):
            if torch.all(layout[:, lf_col] == layout[:, col_]):
                col_count = col_count + 1
            else:
                break
        
        layout[:, :] = 0
        layout[lf_row : y.max() + 1, lf_col : lf_col + col_count] = 1
        return layout, layout_score[lf_row : y.max() + 1, lf_col : lf_col + col_count].mean()

def gen_proposals(layout_score, srow, scol, score_threshold=0.5):
    layout = layout_score > score_threshold
    layout[srow, scol] = 1
    
    y, x = torch.where(layout == 1)
    if torch.all(layout[y.min():y.max() + 1, x.min():x.max()+1] == 1):
        return layout.unsqueeze(0), layout_score[y.min():y.max() + 1, x.min():x.max()+1].mean().unsqueeze(0).log()
    else:
        proposal_1, score_1 = proposal_colspan(copy.deepcopy(layout), layout_score, srow, scol)
        proposal_2, score_2 = proposal_rowspan(copy.deepcopy(layout), layout_score, srow, scol)
        proposal_3, score_3 = proposal_maxcontain(copy.deepcopy(layout), layout_score, srow, scol)
        proposal_4, score_4 = proposal_maxrowspan(copy.deepcopy(layout), layout_score, srow, scol)
        proposal_5, score_5 = proposal_maxcolspan(copy.deepcopy(layout), layout_score, srow, scol)
        proposals = torch.stack([proposal_1, proposal_2, proposal_3, proposal_4, proposal_5], dim=0)
        scores = torch.stack([score_1.log(), score_2.log(), score_3.log(), score_4.log(), score_5.log()], dim=0)
        return proposals, scores
    
def extend_segments(row_segments, rows_es, col_segments, cols_es, cells_spans, layouts, divide_labels):
    batch_size = len(row_segments)
    ext_row_segments = list()
    ext_col_segments = list()
    ext_cells_spans = list()
    ext_layouts = list()
    ext_divide_labels = list()
    for batch_idx in range(batch_size):
        row_segments_pi = row_segments[batch_idx]
        col_segments_pi = col_segments[batch_idx]
        rows_es_pi = rows_es[batch_idx]
        cols_es_pi = cols_es[batch_idx]
        cells_spans_pi = cells_spans[batch_idx]

        ext_row_segments_pi = row_segments_pi + rows_es_pi
        ext_col_segments_pi = col_segments_pi + cols_es_pi

        row_segments_idx = sorted(list(range(len(ext_row_segments_pi))), key=lambda idx: ext_row_segments_pi[idx])
        col_segments_idx = sorted(list(range(len(ext_col_segments_pi))), key=lambda idx: ext_col_segments_pi[idx])

        ext_divide_labels.append(row_segments_idx.index(divide_labels[batch_idx].item()))

        ext_row_segments.append([ext_row_segments_pi[idx] for idx in row_segments_idx])
        ext_col_segments.append([ext_col_segments_pi[idx] for idx in col_segments_idx])

        ext_layouts_pi = np.full((len(ext_row_segments_pi) - 1, len(ext_col_segments_pi) - 1), -1)
        ext_cells_spans_pi = list()
        for cell_idx, cell_span in enumerate(cells_spans_pi):
            l, t, r, b = cell_span
            l = col_segments_idx.index(l)
            r = col_segments_idx.index(r+1) - 1
            t = row_segments_idx.index(t)
            b = row_segments_idx.index(b+1) - 1
            ext_cells_spans_pi.append([l, t, r, b])
            ext_layouts_pi[t:b+1, l:r+1] = cell_idx
        ext_cells_spans.append(ext_cells_spans_pi)
        ext_layouts.append(ext_layouts_pi)

    return ext_row_segments, ext_col_segments, ext_cells_spans, aligned_layouts(ext_layouts, layouts), torch.tensor(ext_divide_labels).to(divide_labels.device)

def aligned_layouts(layouts_list, layouts):
    batch_size = len(layouts_list)
    dtype = layouts.dtype
    device = layouts.device

    max_row_nums = max([l.shape[0] for l in layouts_list])
    max_col_nums = max([l.shape[1] for l in layouts_list])

    aligned_layouts = list()
    for batch_idx in range(batch_size):
        num_rows_pi = layouts_list[batch_idx].shape[0]
        num_cols_pi = layouts_list[batch_idx].shape[1]
        layouts_pi = torch.from_numpy(layouts_list[batch_idx]).to(dtype=dtype, device=device)
        aligned_layouts_pi = F.pad(
            layouts_pi,
            (0, max_col_nums-num_cols_pi, 0, max_row_nums-num_rows_pi),
            mode='constant',
            value=-1
        )
        aligned_layouts.append(aligned_layouts_pi)
    aligned_layouts = torch.stack(aligned_layouts, dim=0)
    return aligned_layouts

def parse_layout(spans, num_rows, num_cols):
    layout = np.full([num_rows, num_cols], -1, dtype=np.int)
    cell_count = 0
    for x1, y1, x2, y2, prob in spans:
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


def parse_cells(layout, row_segments, col_segments):
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
    return cells


def process_layout(score, index):
    layout = torch.full_like(index, -1)
    layout_mask = torch.full_like(index, -1)
    nrow, ncol = score.shape
    for cell_id in range(nrow * ncol):
        if layout_mask.min() != -1:
            break
        crow, ccol = torch.where(layout_mask == layout_mask.min())
        ccol = ccol[crow == crow.min()].min()
        crow = crow.min()
        id = index[crow, ccol]
        h, w = torch.where(index == id)
        if h.shape[0] == 1 or w.shape[0] == 1: 
            layout_mask[h, w] = 1
            layout[h, w] = cell_id
            continue
        else:
            h_min = h.min()
            h_max = h.max()
            w_min = w.min()
            w_max = w.max()
            if torch.all(index[h_min:h_max+1, w_min:w_max+1] == id):
                layout_mask[h_min:h_max+1, w_min:w_max+1] = 1
                layout[h_min:h_max+1, w_min:w_max+1] = cell_id
            else:
                lf_row = crow
                lf_col = ccol
                
                col_mem = -1
                for col_ in range(lf_col, w_max + 1):
                    if index[lf_row, col_] == id:
                        layout_mask[lf_row, col_] = 1
                        layout[lf_row, col_] = cell_id
                        col_mem = col_
                    else:
                        break
                for row_ in range(lf_row + 1, h_max + 1):
                    if torch.all(index[row_, lf_col: col_mem + 1] == id):
                        layout_mask[row_, lf_col: col_mem + 1] = 1
                        layout[row_, lf_col: col_mem + 1] = cell_id
                    else:
                        break
    return layout

def process_layout(score, index, use_score=False, is_merge=True, score_threshold=0.5):
    if use_score:
        if is_merge:
            y, x = torch.where(score < score_threshold)
            index[y, x] = index.max() + 1
        else:
            y, x = torch.where(score < score_threshold)
            index[y, x] = torch.arange(index.max() + 1, index.max() + 1 + len(y)).to(index.device, index.dtype)

    layout = torch.full_like(index, -1)
    layout_mask = torch.full_like(index, -1)
    nrow, ncol = score.shape
    for cell_id in range(max(nrow * ncol, index.max() + 1)):
        if layout_mask.min() != -1:
            break
        crow, ccol = torch.where(layout_mask == layout_mask.min())
        ccol = ccol[crow == crow.min()].min()
        crow = crow.min()
        id = index[crow, ccol]
        h, w = torch.where(index == id)
        if h.shape[0] == 1 or w.shape[0] == 1: 
            layout_mask[h, w] = 1
            layout[h, w] = cell_id
            continue
        else:
            h_min = h.min()
            h_max = h.max()
            w_min = w.min()
            w_max = w.max()
            if torch.all(index[h_min:h_max+1, w_min:w_max+1] == id):
                layout_mask[h_min:h_max+1, w_min:w_max+1] = 1
                layout[h_min:h_max+1, w_min:w_max+1] = cell_id
            else:
                lf_row = crow
                lf_col = ccol
                
                col_mem = -1
                for col_ in range(lf_col, w_max + 1):
                    if index[lf_row, col_] == id:
                        layout_mask[lf_row, col_] = 1
                        layout[lf_row, col_] = cell_id
                        col_mem = col_
                    else:
                        break
                for row_ in range(lf_row + 1, h_max + 1):
                    if torch.all(index[row_, lf_col: col_mem + 1] == id):
                        layout_mask[row_, lf_col: col_mem + 1] = 1
                        layout[row_, lf_col: col_mem + 1] = cell_id
                    else:
                        break
    return layout

def layout2spans(layout):
    rows, cols = layout.shape[-2:]
    cells_span = list()
    for cell_id in range(rows * cols):
        cell_positions = np.argwhere(layout == cell_id)
        if len(cell_positions) == 0:
            continue
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return [cells_span]

def spatial_att_to_spans(spatial_att_weight_pred):
    max_score, max_index = spatial_att_weight_pred.max(dim=0)
    layout = process_layout(max_score, max_index, use_score=True, is_merge=False)
    layout = process_layout(max_score, layout)
    
    layout = layout.cpu().numpy()
    spans = layout2spans(layout)
    return spans


def save_logitmap(filename, logit):
    cv2.imwrite(filename, (logit.sigmoid()*255).cpu().numpy().astype('uint8'))


def draw_spans(dst, src, spans, type):
    image = cv2.imread(src)
    H, W, *_ = image.shape
    for span in spans:
        if type == 'col':
            cv2.rectangle(image, (span[0], 0), (span[1], H), (0, 0, 255), thickness=1)
        elif type == 'row':
            cv2.rectangle(image, (0, span[0]), (W, span[1]), (0, 0, 255), thickness=1)
    cv2.imwrite(dst, image)


