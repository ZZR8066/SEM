import torch
from torch import nn
from torch.nn import functional as F
from .sa import SALayer
from libs.utils.metric import cal_cls_acc


def align_segments_feat(segments_feat):
    dtype = segments_feat[0].dtype
    device = segments_feat[0].device
    batch_size = len(segments_feat)
    max_segment_nums = max([item.shape[1] for item in segments_feat])
    aligned_segments_feat = list()
    masks = torch.zeros([batch_size, max_segment_nums], dtype=dtype, device=device)
    
    for batch_idx in range(batch_size):
        cur_segment_nums = segments_feat[batch_idx].shape[1]
        masks[batch_idx, :cur_segment_nums] = 1
        aligned_segments_feat.append(
            F.pad(
                segments_feat[batch_idx],
                (0, max_segment_nums - cur_segment_nums, 0, 0),
                mode='constant',
                value=0
            )
        )
    aligned_segments_feat = torch.stack(aligned_segments_feat, dim=0)
    return aligned_segments_feat, masks


class HeadBodyDividePredictor(nn.Module):
    def __init__(self, in_dim, head_nums, scale=1):
        super().__init__()
        self.in_dim = in_dim
        self.scale = scale
        self.fusion_layer = SALayer(in_dim, in_dim, head_nums)
        self.classifier= nn.Conv1d(in_dim, 1, 1, 1, 0)

    def forward(self, feats, segments, divide_labels=None):
        segments = [[int(subitem * self.scale) for subitem in item] for item in segments]
        segments_feat = [feats_pi[:, segments_pi] for feats_pi, segments_pi in zip(feats, segments)]
        aligned_segments_feat, masks = align_segments_feat(segments_feat)
        aligned_segments_feat = self.fusion_layer(aligned_segments_feat, masks)
        divide_logits = self.classifier(aligned_segments_feat).squeeze(1)
        divide_logits = divide_logits - (1 - masks) * 1e8
        divide_preds = torch.argmax(divide_logits, dim=1)
        
        result_info = dict()
        ext_info = dict()
        if self.training:
            result_info['divide_loss'] = F.cross_entropy(divide_logits, divide_labels)
            correct_nums, total_nums = cal_cls_acc(divide_preds, divide_labels)
            if total_nums != 0:
                result_info['divide_acc'] = correct_nums / total_nums
        
        divide_preds = divide_preds.detach().cpu().tolist()
        return divide_preds, result_info, ext_info
