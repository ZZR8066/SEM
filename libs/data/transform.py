import math
import torch
import random
import numpy as np
from torchvision.transforms import functional as F
from libs.utils.format_translate import table_to_latex
from .utils import extract_fg_bg_spans, cal_cell_spans


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data

class TableToLabel:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, image, table=None):
        if table is None:
            return image, None, None
        latex = table_to_latex(table) # image.size = (w, h)
        cls_label = self.vocab.words_to_ids(latex)
        return image, table, cls_label

class CalRowColSpans:
    def __call__(self, image, table=None, cls_label=None):
        if table is None:
            return image, table, None, None, None, None, None
        image_size = (image.width, image.height)
        rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span = extract_fg_bg_spans(table, image_size)
        return image, table, cls_label, rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span

class CalCellSpans:
    def __call__(self, image, table=None, cls_label=None, rows_fg_span=None, rows_bg_span=None, cols_fg_span=None, cols_bg_span=None):
        if table is not None:
            cells_span = cal_cell_spans(table)
        else:
            cells_span = None
        return image, table, cls_label, rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span, cells_span

class CalHeadBodyDivide:
    def __call__(self, image, table=None, cls_label=None, rows_fg_span=None, rows_bg_span=None, cols_fg_span=None, cols_bg_span=None, cells_span=None):
        if table is None:
            divide = None
        else:
            head_rows = table['head_rows']
            divide = len(head_rows)
        return image, table, cls_label, rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span, cells_span, divide

class ToTensor:
    def __call__(self, image, table=None, cls_label=None, rows_fg_span=None, rows_bg_span=None, cols_fg_span=None, cols_bg_span=None, cells_span=None, divide=None):
        image = F.to_tensor(image)
        if cls_label is not None:
            cls_label = torch.tensor(cls_label, dtype=torch.long)
        return image, table, cls_label, rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span, cells_span, divide

class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image, table=None, cls_label=None, rows_fg_span=None, rows_bg_span=None, cols_fg_span=None, cols_bg_span=None, cells_span=None, divide=None):
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image, table, cls_label, rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span, cells_span, divide
