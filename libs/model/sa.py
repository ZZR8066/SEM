import math
import torch
from torch import nn


class SALayer(nn.Module):
    def __init__(self, in_dim, att_dim, head_nums):
        super().__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.head_nums = head_nums

        assert self.in_dim % self.head_nums == 0

        self.key_layer = nn.Conv1d(self.in_dim, self.att_dim * self.head_nums, 1, 1, 0)
        self.query_layer = nn.Conv1d(self.in_dim, self.att_dim * self.head_nums, 1, 1, 0)
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


