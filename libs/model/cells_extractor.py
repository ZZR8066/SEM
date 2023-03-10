import torch
import math
from torch import nn
from torch.nn import functional as F
from .extractor import RoiPosFeatExtraxtor


class SALayer(nn.Module):
    def __init__(self, in_dim, att_dim, head_nums):
        super().__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.head_nums = head_nums

        assert self.in_dim % self.head_nums == 0

        self.key_layer = nn.Conv1d(self.in_dim, self.att_dim, 1, 1, 0)
        self.query_layer = nn.Conv1d(self.in_dim, self.att_dim, 1, 1, 0)
        self.value_layer = nn.Conv1d(self.in_dim, self.in_dim, 1, 1, 0)
        self.scale = 1 / math.sqrt(self.att_dim)

    def forward(self, feats, masks=None):
        bs, c, n = feats.shape
        keys = self.key_layer(feats).reshape(bs, -1, self.head_nums, n)
        querys = self.query_layer(feats).reshape(bs, -1, self.head_nums, n)
        values = self.value_layer(feats).reshape(bs, -1, self.head_nums, n)

        logits = torch.einsum('bchk,bchq->bhkq', keys, querys) * self.scale
        if masks is not None:
            logits = logits - (1 - masks[:, None, :, None]) * 1e8
        weights = torch.softmax(logits, dim=2)

        new_feats = torch.einsum('bchk,bhkq->bchq', values, weights)
        new_feats = new_feats.reshape(bs, -1, n)
        return new_feats + feats


def gen_cells_bbox(row_segments, col_segments, device):
    cells_bbox = list()
    for row_segments_pi, col_segments_pi in zip(row_segments, col_segments):
        num_rows = len(row_segments_pi) - 1
        num_cols = len(col_segments_pi) - 1
        cells_bbox_pi = list()
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                bbox = [
                    col_segments_pi[col_idx],
                    row_segments_pi[row_idx],
                    col_segments_pi[col_idx + 1],
                    row_segments_pi[row_idx + 1]
                ]
                cells_bbox_pi.append(bbox)
        cells_bbox_pi = torch.tensor(cells_bbox_pi, dtype=torch.float, device=device)
        cells_bbox.append(cells_bbox_pi)
    return cells_bbox


def align_cells_feat(cells_feat, num_rows, num_cols):
    batch_size = len(cells_feat)
    dtype = cells_feat[0].dtype
    device = cells_feat[0].device

    max_row_nums = max(num_rows)
    max_col_nums = max(num_cols)

    aligned_cells_feat = list()
    masks = torch.zeros([batch_size, max_row_nums, max_col_nums], dtype=dtype, device=device)
    for batch_idx in range(batch_size):
        num_rows_pi = num_rows[batch_idx]
        num_cols_pi = num_cols[batch_idx]
        cells_feat_pi = cells_feat[batch_idx]
        cells_feat_pi = cells_feat_pi.transpose(0, 1).reshape(-1, num_rows_pi, num_cols_pi)
        aligned_cells_feat_pi = F.pad(
            cells_feat_pi,
            (0, max_col_nums - num_cols_pi, 0, max_row_nums - num_rows_pi, 0, 0),
            mode='constant',
            value=0
        )
        aligned_cells_feat.append(aligned_cells_feat_pi)

        masks[batch_idx, :num_rows_pi, :num_cols_pi] = 1
    aligned_cells_feat = torch.stack(aligned_cells_feat, dim=0)
    return aligned_cells_feat, masks


class CellsExtractor(nn.Module):
    def __init__(self, in_dim, cell_dim, heads, head_nums, pool_size, scale=1):
        super().__init__()
        self.in_dim = in_dim
        self.cell_dim = cell_dim
        self.pool_size = pool_size
        self.scale = scale
        self.box_feat_extractor = RoiPosFeatExtraxtor(
            self.scale,
            self.pool_size,
            self.in_dim,
            self.cell_dim
        )
        self.heads = heads
        self.row_sas = nn.ModuleList()
        self.col_sas = nn.ModuleList()
        for _ in range(self.heads):
            self.row_sas.append(SALayer(cell_dim, cell_dim, head_nums))
            self.col_sas.append(SALayer(cell_dim, cell_dim, head_nums))


    def forward(self, feats, row_segments, col_segments, img_sizes):
        device = feats.device
        num_rows = [len(row_segments_pi) - 1 for row_segments_pi in row_segments]
        num_cols = [len(col_segments_pi) - 1 for col_segments_pi in col_segments]

        cells_bbox = gen_cells_bbox(row_segments, col_segments, device)
        cells_feat = self.box_feat_extractor(feats, cells_bbox, img_sizes)

        aligned_cells_feat, masks = align_cells_feat(cells_feat, num_rows, num_cols)
        
        bs, c, nr, nc = aligned_cells_feat.shape

        for idx in range(self.heads):
            col_cells_feat = aligned_cells_feat.permute(0, 2, 1, 3).contiguous().reshape(bs * nr, c, nc)
            col_masks = masks.reshape(bs * nr, nc)
            col_cells_feat = self.col_sas[idx](col_cells_feat, col_masks) # self-attention
            aligned_cells_feat = col_cells_feat.reshape(bs, nr, c, nc).permute(0, 2, 1, 3).contiguous()

            row_cells_feat = aligned_cells_feat.permute(0, 3, 1, 2).contiguous().reshape(bs * nc, c, nr)
            row_masks = masks.transpose(1, 2).reshape(bs * nc, nr)
            row_cells_feat = self.row_sas[idx](row_cells_feat, row_masks) # self-attention
            aligned_cells_feat = row_cells_feat.reshape(bs, nc, c, nr).permute(0, 2, 3, 1).contiguous()
        
        return aligned_cells_feat, masks
