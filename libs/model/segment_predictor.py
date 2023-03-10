import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
from libs.utils.metric import cal_segment_pr
from .utils import draw_spans, save_logitmap

def cal_segments(cls_probs, spans, scale=1.0):
    segments = list()
    for span in spans:
        span_cls_probs = cls_probs[int(span[0] * scale): int(span[1] * scale)]
        segment = torch.argmax(span_cls_probs).item() + int(span[0] * scale)
        segments.append(segment)
    segments = [int(item/scale) for item in segments]
    return segments


def cal_spans(cls_probs, threshold=0.5):
    ids = (cls_probs > threshold).long().tolist()
    spans = list()
    for idx, id in enumerate(ids):
        if id == 1:
            if (idx == 0) or (ids[idx-1] != 1):
                spans.append([idx, idx+1])
            else:
                spans[-1][1] = idx + 1
    return spans
# draw_spans('row_segment_spans.png', 'row_segment.png', spans, 'row')

def cls_logits_to_segments(segments_logit, masks, type, spans=None, scale=1, threshold=0.5):
    if type == 'col':
        cls_probs = segments_logit.squeeze(1).sigmoid().mean(dim=1)
        lengths = [int(mask[0, :].sum().item()) for mask in masks]
    else:
        cls_probs = segments_logit.squeeze(1).sigmoid().mean(dim=2)
        lengths = [int(mask[:, 0].sum().item()) for mask in masks]

    batch_size = cls_probs.shape[0]
    segments = list()
    for batch_idx in range(batch_size):
        length = lengths[batch_idx]
        if spans is None:
            spans_pi = cal_spans(cls_probs[batch_idx, :length], threshold)
            if len(spans_pi) <= 2:
                spans_pi = [[0, 1], [length-1, length]]
        else:
            spans_pi = spans[batch_idx]
        segments_pi = cal_segments(cls_probs[batch_idx, :length], spans_pi, scale)
        segments.append(segments_pi)
    return segments, cls_probs, lengths


def cal_ext_segments(cls_probs, lengths, bg_spans, scale=1, threshold=0.5):
    """
    寻找假阳性. 在bg_spans(非line区域,即文字区域)中寻找预测概率最大, 且大于threshold的行.
    """
    batch_size = cls_probs.shape[0]
    ext_segments = list()
    for batch_idx in range(batch_size):
        length = lengths[batch_idx]
        ext_segments_pi = cal_segments(cls_probs[batch_idx, :length], bg_spans[batch_idx], scale)
        ext_segments_pi = [segment for segment in ext_segments_pi if cls_probs[batch_idx, segment] > threshold]
        ext_segments.append(ext_segments_pi)
    return ext_segments
    

def gen_masks(sizes, scale, device):
    batch_size = len(sizes)
    max_size = [int(max(item) * scale) for item in zip(*sizes)]
    masks = torch.zeros([batch_size, *max_size], dtype=torch.float, device=device)
    for batch_idx in range(batch_size):
        masks[batch_idx, :sizes[batch_idx][0], :sizes[batch_idx][1]] = 1.
    return masks


def gen_targets(sizes, scale, device, fg_spans, bg_spans, type):
    batch_size = len(sizes)
    max_size = [int(max(item) * scale) for item in zip(*sizes)]
    targets = torch.zeros([batch_size, *max_size], dtype=torch.float, device=device)
    for batch_idx, fg_spans_pb in enumerate(fg_spans):
        if type == 'col':
            for fg_spans_pi in fg_spans_pb:
                targets[batch_idx, :, int(fg_spans_pi[0] * scale) : int(fg_spans_pi[1] * scale)] = 1.
        else:
            for fg_spans_pi in fg_spans_pb:
                targets[batch_idx, int(fg_spans_pi[0] * scale) : int(fg_spans_pi[1] * scale), :] = 1.
    return targets


class SegmentPredictor(nn.Module):
    def __init__(self, in_dim, scale=1, threshold=0.5, type=None):
        super().__init__()
        self.scale = scale
        self.in_dim = in_dim
        assert type in ['col', 'row']
        self.type = type
        self.threshold = threshold
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        )

    def forward(self, feats, images_size, fg_spans=None, bg_spans=None):
        batch_size = feats.shape[0]
        images_size = [image_size[::-1] for image_size in images_size]
        segments_logit = self.convs(feats)
        masks = gen_masks(images_size, self.scale, feats.device)
        # save_logitmap('row_segment.png', segments_logit[0][0])
        result_info = dict()
        ext_info = dict()

        if self.training:
            targets = gen_targets(images_size, self.scale, feats.device, fg_spans, bg_spans, self.type)
            segments_loss = F.binary_cross_entropy_with_logits(
                segments_logit,
                targets.unsqueeze(1),
                reduction='none'
            )
            segments_loss = (segments_loss * masks[:, None, :, :]).sum() / targets.sum()
            result_info['segments_loss'] = segments_loss
            
            pred_segments, cls_probs, lengths = cls_logits_to_segments(segments_logit, masks, self.type, spans=None, scale=self.scale, threshold=self.threshold)
            correct_nums, segment_nums, span_nums = cal_segment_pr(pred_segments, fg_spans, bg_spans)
            if segment_nums != 0:
                result_info['precision'] = correct_nums/segment_nums
            if span_nums != 0:
                result_info['recall'] = correct_nums/span_nums
            ext_segments = cal_ext_segments(cls_probs, lengths, bg_spans, self.scale, self.threshold)
            ext_info['ext_segments'] = ext_segments

        pred_segments, *_ = cls_logits_to_segments(segments_logit, masks, self.type, spans=fg_spans, scale=self.scale, threshold=self.threshold)
        return pred_segments, result_info, ext_info
