import os
import cv2
import json
import copy
import tqdm
import numpy as np
import fitz
from .extract_table_lines import extract_fg_bg_spans


def get_paths(root_dir, sub_names, names_path, exts, val=None):
    # Check the existence of directories
    assert os.path.isdir(root_dir)

    with open(names_path, "r") as f:
        names = f.readlines()
        names = [name.strip() for name in names]
    
    # TODO: sub_dirs redundancy
    sub_dirs = []
    for sub_name in sub_names:
        sub_dir = os.path.join(root_dir, sub_name)
        assert os.path.isdir(sub_dir), '"%s" is not dir.' % sub_dir
        sub_dirs.append(sub_dir)
    
    paths = []
    names = names[:val]
    for name in tqdm.tqdm(names):
        sub_paths = []
        for sub_dir, ext in zip(sub_dirs, exts):
            sub_path = os.path.join(sub_dir, name + ext)
            assert os.path.exists(sub_path), print('%s is not exist' % sub_path)
            sub_paths.append(sub_path)
        paths.append(sub_paths)
        
    return paths


def get_sub_paths(root_dir, sub_names, exts, val=None):
    # Check the existence of directories
    assert os.path.isdir(root_dir)
    # TODO: sub_dirs redundancy
    sub_dirs = []
    for sub_name in sub_names:
        sub_dir = os.path.join(root_dir, sub_name)
        assert os.path.isdir(sub_dir), '"%s" is not dir.' % sub_dir
        sub_dirs.append(sub_dir)

    paths = []
    d = os.listdir(sub_dirs[0])[:val]
    for file_name in tqdm.tqdm(d):
        sub_paths = [os.path.join(sub_dirs[0], file_name)]
        name = os.path.splitext(file_name)[0]
        for sub_name, ext in zip(sub_names[1:], exts[1:]):
            sub_path = os.path.join(root_dir, sub_name, name + ext)
            assert os.path.exists(sub_path)
            sub_paths.append(sub_path)
        paths.append(sub_paths)
        
    return paths


def cal_wer(label, rec):
    dist_mat = np.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
    dist_mat[0, :] = range(len(rec) + 1)
    dist_mat[:, 0] = range(len(label) + 1)

    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i - 1, j - 1] + (label[i - 1] != rec[j - 1])
            ins_score = dist_mat[i, j - 1] + 1
            del_score = dist_mat[i - 1, j] + 1
            dist_mat[i, j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]

    return 1 - dist / len(label)


def visualize(img_path, chunks, structures):
    image = cv2.imread(img_path)
    for chunk in chunks:
        x1, x2, y1, y2 = chunk["pos"]
        transcript = chunk["text"]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
        cv2.putText(image, ''.join(transcript), (int(x1), int(max(0, y1-1))), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0 , 0, 255), 1)
    return image


def visualize_table(img_path, output_dir, table):
    img = cv2.imread(img_path)
    for cell in table['cells']:
        x1, y1, x2, y2 = cell['bbox']
        transcript = cell['transcript']
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
        cv2.putText(img, ''.join(transcript), (int(x1), int(max(0, y1-1))), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0 , 0, 255), 1)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)


def crop_pdf(path, output_dir, zoom_x = 2.0, zoom_y = 2.0, rotate=0, expand=10, y_fix=.0):
    '''
        path:[pdf_path, chunk_path]
        crop table region in pdf
        save pdf_name.png
        return list[x1, x2, y1, y2], [str]. note these are corresponding to crop pdf
    '''
    # load data
    with open(path[1], 'r') as f:
        chunks = json.load(f)['chunks']
    doc = fitz.open(path[0])
    pdf_name = os.path.splitext(os.path.basename(path[0]))[0]
    assert doc.pageCount == 1, print(pdf_name, ' has more than 1 page!')

    # transfer pdf to img
    trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
    pm = doc[0].getPixmap(matrix=trans, alpha=False)
    pm.writePNG(os.path.join(output_dir, '%s.png' % pdf_name))

    # crop table region
    pdf_img = cv2.imread(os.path.join(output_dir, '%s.png' % pdf_name))
    h, w, *_ = pdf_img.shape
    positions = []
    transcripts = []
    for chunk in chunks:
        positions.append([chunk['pos'][0], chunk['pos'][1], chunk['pos'][3], chunk['pos'][2]]) # x1, x2, y2, y1
        transcripts.append(chunk["text"])

    # the last chunk transcrip is repeated
    transcripts[-1] = transcripts[-1][:-1]

    positions = np.array(positions)
    positions[:, :2] *= zoom_x
    positions[:, 2:] = h - positions[:, 2:] * zoom_y
    x_min = int(max(0, positions[:, :2].min() - expand))
    y_min = int(max(0, positions[:, 2:].min() - expand))
    x_max = int(min(w, positions[:, :2].max() + expand))
    y_max = int(min(h, positions[:, 2:].max() + expand))

    img_crop = pdf_img[y_min:y_max, x_min:x_max]
    cv2.imwrite(os.path.join(output_dir, '%s.png' % pdf_name), img_crop)

    positions[:, :2] = np.clip(positions[:, :2] - x_min, 0, w)
    positions[:, 2] -= y_fix * zoom_y
    positions[:, 2:] = np.clip(positions[:, 2:] - y_min, 0, h)
    return positions, transcripts


def crop_cells(img_path, output_dir, info, expand=10):
    cells = info['cells']
    img = cv2.imread(img_path)
    h, w, *_ = img.shape
    bboxes = [cell['bbox'] for cell in cells if 'bbox' in cell.keys()]            
    bboxes = np.array(bboxes)
    x_min = int(max(bboxes[:, 0].min() - expand, 0))
    y_min = int(max(bboxes[:, 1].min() - expand, 0))
    x_max = int(min(bboxes[:, 2].max() + expand, w))
    y_max = int(min(bboxes[:, 3].max() + expand, h))
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png'), img[y_min:y_max, x_min:x_max])
    
    # refine cell bbox
    new_cells = []
    for cell in cells:
        if 'bbox' not in cell.keys():
            new_cells.append(cell)
        else:
            cell['bbox'][0] = max(0, cell['bbox'][0] - x_min)
            cell['bbox'][1] = max(0, cell['bbox'][1] - y_min)
            cell['bbox'][2] = max(0, cell['bbox'][2] - x_min)
            cell['bbox'][3] = max(0, cell['bbox'][3] - y_min)
            segmentation = cell['segmentation']
            cell['segmentation'] = [[[pt[0] - x_min, pt[1] - y_min] for pt in contour] for contour in segmentation]
            new_cells.append(cell)
    info['cells'] = new_cells


def visualize_ocr(img_path, output_dir, positions, transcripts):
    img = cv2.imread(img_path)
    for position, transcript in zip(positions, transcripts):
        x1, x2, y1, y2 = position
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
        cv2.putText(img, transcript, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '_ocr.png'), img)


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


def visualize_cell(img_path, output_dir, table):
    def spans2lines(spans):
        lines = []
        lines.append(spans[0][0])
        for span in spans[1:-1]:
            t1, t2 = span
            lines.append(int((t1 + t2) / 2))
        lines.append(spans[-1][-1])
        return lines

    img = cv2.imread(img_path)

    # draw table lines
    rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span, cells_span = extract_fg_bg_spans(table, img.shape[::-1][-2:])
    row_lines = spans2lines(rows_fg_span)
    col_lines = spans2lines(cols_fg_span)
    for span in cells_span:
        x1, y1, x2, y2 = span
        cv2.rectangle(img, (int(col_lines[x1]), int(row_lines[y1])), (int(col_lines[x2 + 1]), int(row_lines[y2 + 1])), (0, 0, 255), 2)

    # draw ocr results
    for cell in table['cells']:
        if 'bbox' not in cell.keys():
            continue
        x1, y1, x2, y2 = cell['bbox']
        transcript = cell['transcript']
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.putText(img, ''.join(transcript), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png'), img)


def match_cells(path, positions, transcripts, k=16, start=0.333, stop=0.1, stop_percent=0.3, gap=0.25):
    '''
        path: [pdf_path, chunk_path, structure_path]
        positions: [x1, x2, y1, y2], 
        transcripts: [str]
        retrun dict(
            'layout':np.array()
            'bbox':[x1, y1, x2, y2]
            'transcript: str
            'head_rows':[]
            'body_rows':[]
        )
    '''
    # load data
    with open(path[2], 'r') as f:
        cells = json.load(f)['cells']
    
    # first sort cells from left to right, from top to down
    cells_pos = [] # xl1, yl1, xl2, yl2
    contents = []
    for cell in cells:
        cells_pos.append([cell['start_col'], cell['start_row'], cell['end_col'], cell['end_row']])
        contents.append(' '.join(cell['content']))
    
    # sorted cells from left to right, from top to down
    sorted_idx = sorted(list(range(len(cells_pos))), key=lambda idx: cells_pos[idx][0] + 1e6 * cells_pos[idx][1])
    cells_pos = [cells_pos[idx] for idx in sorted_idx]
    contents = [contents[idx] for idx in sorted_idx]

    # layout
    n_row = np.array(cells_pos)[:, 3].max() + 1
    n_col = np.array(cells_pos)[:, 2].max() + 1
    layout = np.full((n_row, n_col), -1)

    # head_rows & body_rows
    head_rows = list(range((np.array(cells_pos)[np.array(cells_pos)[:,1] == 0][:, 3] - np.array(cells_pos)[np.array(cells_pos)[:,1] == 0][:, 1]).max() + 1))
    body_rows = list(range((np.array(cells_pos)[np.array(cells_pos)[:,1] == 0][:, 3] - np.array(cells_pos)[np.array(cells_pos)[:,1] == 0][:, 1]).max() + 1, n_row))

    lt = [-1, -1]
    cells = []
    valid_idx = list(range(len(transcripts)))

    # init start/end index of ocr results
    start_content = ''
    for content in contents:
        if len(content) > 0:
            start_content = content
            break
    try:
        start_index = [cal_wer(start_content, transcript) > start for transcript in transcripts[:k]].index(True)
    except:
        start_index = 0

    end_content = ''
    for content in contents[::-1]:
        if len(content) > 0:
            end_content = content
            break
    try:
        end_index = [cal_wer(end_content, transcript) > start for transcript in transcripts[::-1][:k]].index(True)
    except:
        end_index = 0

    valid_idx = valid_idx[start_index:] if end_index == 0 else valid_idx[start_index: -end_index]

    assert len(contents) >= len(valid_idx), print('OCR Results Have Error')

    stop_counts = 0
    for index, (cell_pos, content) in enumerate(zip(cells_pos, contents)):
        # confirm the cell pos is increase
        assert cell_pos[0] > lt[0] or cell_pos[1] > lt[1], print('Sorted Cells Have Error')
        lt = cell_pos[:2]

        xl1, yl1, xl2, yl2 = cell_pos
        layout[yl1:yl2+1, xl1:xl2+1] = index

        if len(content) == 0:
            cells.append(dict(transcript=[]))
        else:
            is_completed = False
            bboxes_list = [positions[valid_idx[0]]]
            transcripts_list = [transcripts[valid_idx[0]]]
            valid_idx.pop(0)
            wer_last = cal_wer(content, ' '.join(transcripts_list))
            if wer_last < stop:
                bboxes_list = np.array(bboxes_list)
                x1 = int(bboxes_list[:, :2].min())
                x2 = int(bboxes_list[:, :2].max())
                y1 = int(bboxes_list[:, 2:].min())
                y2 = int(bboxes_list[:, 2:].max())
                cells.append(dict(transcript=list(content), bbox=[x1, y1, x2, y2], segmentation=[[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]))
                stop_counts += 1
                continue
            for idx in valid_idx[:k]:
                if content == ' '.join(transcripts_list):
                    bboxes_list = np.array(bboxes_list)
                    x1 = int(bboxes_list[:, :2].min())
                    x2 = int(bboxes_list[:, :2].max())
                    y1 = int(bboxes_list[:, 2:].min())
                    y2 = int(bboxes_list[:, 2:].max())
                    cells.append(dict(transcript=list(content), bbox=[x1, y1, x2, y2], segmentation=[[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]))
                    is_completed = True
                    break
                else:
                    cur_trans = copy.deepcopy(transcripts_list)
                    cur_trans.append(transcripts[idx])
                    wer = cal_wer(content, ' '.join(cur_trans))
                    # if add new str, and wer is not increase a lot, it should not be added in
                    if wer < wer_last + gap:
                        continue
                    else:
                        transcripts_list.append(transcripts[idx])
                        bboxes_list.append(positions[idx])
                        valid_idx.pop(valid_idx.index(idx))
                        if wer == 1.0:
                            break
                        else:
                            wer_last = wer
            if not is_completed:
                bboxes_list = np.array(bboxes_list)
                x1 = int(bboxes_list[:, :2].min())
                x2 = int(bboxes_list[:, :2].max())
                y1 = int(bboxes_list[:, 2:].min())
                y2 = int(bboxes_list[:, 2:].max())
                cells.append(dict(transcript=list(content), bbox=[x1, y1, x2, y2], segmentation=[[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]))

    assert stop_counts / len(contents) < stop_percent, print('This Table Has Many Error Match with OCR Results')
    assert layout.min() == 0, print('This Table Layout is not Completely Resolved')
    return dict(
        layout=layout,
        cells=cells,
        head_rows=head_rows,
        body_rows=body_rows,
    )


def extract_ocr(path, positions, transcripts, k=16, start=0.333):
    '''
        path: [pdf_path, chunk_path, structure_path]
        positions: [x1, x2, y1, y2], 
        transcripts: [ ]
        retrun dict(
            'cells':{
                'bbox':[x1, y1, x2, y2]
                'transcript: []
            }
        )
    '''
    # load data
    with open(path[2], 'r') as f:
        cells = json.load(f)['cells']
    
    # first sort cells from left to right, from top to down
    cells_pos = [] # xl1, yl1, xl2, yl2
    contents = []
    for cell in cells:
        cells_pos.append([cell['start_col'], cell['start_row'], cell['end_col'], cell['end_row']])
        contents.append(' '.join(cell['content']))
    
    # sorted cells from left to right, from top to down
    sorted_idx = sorted(list(range(len(cells_pos))), key=lambda idx: cells_pos[idx][0] + 1e6 * cells_pos[idx][1])
    cells_pos = [cells_pos[idx] for idx in sorted_idx]
    contents = [contents[idx] for idx in sorted_idx]

    # init start/end index, condition is the first/last index must not over split, and wer should be larger than start threshold
    valid_idx = list(range(len(transcripts)))
    start_content = ''
    for content in contents:
        if len(content) > 0:
            start_content = content
            break
    try:
        start_index = [cal_wer(start_content, transcript) > start for transcript in transcripts[:k]].index(True)
    except:
        start_index = 0
    
    end_content = ''
    for content in contents[::-1]:
        if len(content) > 0:
            end_content = content
            break
    try:
        end_index = [cal_wer(end_content, transcript) > start for transcript in transcripts[::-1][:k]].index(True)
    except:
        end_index = 0

    valid_idx = valid_idx[start_index:] if end_index == 0 else valid_idx[start_index: -end_index]

    cells = []
    for idx in valid_idx:
        x1, x2, y1, y2 = positions[idx].astype('int').tolist()
        cells.append(dict(transcript=list(transcripts[idx]), bbox=[x1, y1, x2, y2], segmentation=[[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]))

    return dict(
        cells=cells
    )


def refine_table(table, img_path, output_dir, expand=10):
    cells = table['cells']
    bboxes = [cell['bbox'] for cell in table['cells'] if 'bbox' in cell.keys()]
    bboxes = np.array(bboxes)
    img = cv2.imread(img_path)
    h, w, *_ = img.shape
    x1 = int(max(0, bboxes[:, 0].min() - expand))
    y1 = int(max(0, bboxes[:, 1].min() - expand))
    x2 = int(min(w, bboxes[:, 2].max() + expand))
    y2 = int(min(h, bboxes[:, 3].max() + expand))
    # refine cells
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2] - x1, 0, 1e6)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2] - y1, 0, 1e6)
    bboxes = bboxes.tolist()
    for cell, bbox in zip(cells, bboxes):
        cell['bbox'] = bbox

    img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)
    table['image_path'] = os.path.join(output_dir, os.path.basename(img_path))
    return table