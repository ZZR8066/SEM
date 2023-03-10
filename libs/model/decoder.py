import math
from numpy.core.fromnumeric import argmax
import torch
from torch import nn
from torch._C import device, dtype, layout
from torch.nn import functional as F
from torch.nn.functional import cross_entropy, embedding
from torch.nn.modules import loss
from torch.nn.modules.activation import Tanh
from libs.utils.metric import CellMergeAcc, AccMetric
from .utils import gen_proposals


class ImageAttention(nn.Module):
    def __init__(self, key_dim, query_dim, cover_kernel):
        super().__init__()
        self.query_transform = nn.Linear(query_dim, key_dim)
        self.weight_transform = nn.Conv2d(1, key_dim, cover_kernel, 1, padding=cover_kernel // 2)
        self.cum_weight_transform = nn.Conv2d(1, key_dim, cover_kernel, 1, padding=cover_kernel // 2)
        self.logit_transform = nn.Conv2d(key_dim, 1, 1, 1, 0)
    
    def forward(self, key, key_mask, query, spatial_att_weight, cum_spatial_att_weight, value, state, layouts=None, layouts_cum=None, spatial_att_weight_scores=None):
        query = self.query_transform(query)
        weight_query = self.weight_transform(spatial_att_weight)
        cum_weight_query = self.cum_weight_transform(cum_spatial_att_weight)
        fusion = key + query[:, :, None, None] + weight_query + cum_weight_query
        # cal new_spatial_att_logit
        new_spatial_att_logit = self.logit_transform(torch.tanh(fusion)) 
        # cal new_spatial_att_weight
        new_spatial_att_weight = new_spatial_att_logit - (1 - key_mask) * 1e8
        bs, _, h, w = new_spatial_att_weight.shape
        new_spatial_att_weight = new_spatial_att_weight.reshape(bs, h * w)
        new_spatial_att_weight = torch.softmax(new_spatial_att_weight, dim=1).reshape(bs, 1, h, w) 
        # cal new_cum_spatial_att_weight
        if self.training:
            outputs = list()
            for (value_pi, layout) in zip(value, layouts):
                h, w = torch.where(layout == 1.)
                if len(h) == 0 or len(w) == 0:
                    outputs.append(torch.zeros_like(query[0]))
                else:
                    outputs.append(value_pi[:, h, w].mean(-1))
            outputs = torch.stack(outputs, dim=0)
            new_cum_spatial_att_weight = torch.clamp(layouts.unsqueeze(1).float() + cum_spatial_att_weight, max=1.)
            return state, outputs, new_spatial_att_logit, new_spatial_att_weight, new_cum_spatial_att_weight, None, None
        else:
            state_list = list()
            outputs_list = list()
            scores_list = list()
            proposals_list = list()
            new_spatial_att_weight_list = list()
            new_cum_spatial_att_weight_list = list()
            layouts_pred = new_spatial_att_logit.squeeze(1).sigmoid()
            for idx, (value_pi, state_pi, layout) in enumerate(zip(value, state, layouts_pred)):
                if cum_spatial_att_weight[idx].min() == 1:
                    state_list.append(state_pi)
                    outputs_list.append(torch.zeros_like(query[0]))
                    proposals_list.append(torch.cat((layouts_cum[idx], torch.zeros_like(layout.unsqueeze(0))), dim=0))
                    scores_list.append(spatial_att_weight_scores[idx])
                    new_spatial_att_weight_list.append(new_spatial_att_weight[idx])
                    new_cum_spatial_att_weight_list.append(cum_spatial_att_weight[idx])
                else:
                    srow, scol = torch.where(cum_spatial_att_weight[idx].squeeze(0) == cum_spatial_att_weight[idx].squeeze(0).min())
                    scol = scol[srow == srow.min()].min()
                    srow = srow.min()
                    proposals, scores = gen_proposals(layout, srow, scol, score_threshold=0.5)
                    scores = scores + spatial_att_weight_scores[idx]
                    for s in scores:
                        scores_list.append(s)
                    for p in proposals:
                        proposals_list.append(torch.cat((layouts_cum[idx], p.unsqueeze(0)), dim=0))
                        h, w = torch.where(p == 1.)                
                        outputs_list.append(value_pi[:, h, w].mean(-1))
                        state_list.append(state_pi)
                        new_spatial_att_weight_list.append(new_spatial_att_weight[idx])
                        new_cum_spatial_att_weight_list.append(torch.clamp(cum_spatial_att_weight[idx] + p.unsqueeze(0), max=1.))
                state_list = torch.stack(state_list, dim=0)
                proposals_list = torch.stack(proposals_list, dim=0)
                scores_list = torch.stack(scores_list, dim=0)
                outputs_list = torch.stack(outputs_list, dim=0)
                new_spatial_att_weight_list = torch.stack(new_spatial_att_weight_list, dim=0)
                new_cum_spatial_att_weight_list = torch.stack(new_cum_spatial_att_weight_list, dim=0)
                sorted_scores, sorted_idxes = torch.sort(scores_list, dim=0, descending=True)
                sorted_scores = sorted_scores[:6]
                sorted_idxes = sorted_idxes[:6]
                proposals = proposals_list[sorted_idxes]
                new_spatial_att_weight = new_spatial_att_weight_list[sorted_idxes]
                new_cum_spatial_att_weight = new_cum_spatial_att_weight_list[sorted_idxes]
                outputs = outputs_list[sorted_idxes]
                state = state_list[sorted_idxes]
                return state, outputs, new_spatial_att_logit, new_spatial_att_weight, new_cum_spatial_att_weight, proposals, sorted_scores



class Decoder(nn.Module):
    def __init__(self, vocab, embed_dim, feat_dim, lm_state_dim, proj_dim, cover_kernel, att_threshold, spatial_att_logit_loss_wight):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.lm_state_dim = lm_state_dim
        self.proj_dim = proj_dim
        self.cover_kernel = cover_kernel
        self.att_threshold = att_threshold
        self.spatial_att_logit_loss_wight = spatial_att_logit_loss_wight
        self.feat_projection = nn.Conv2d(self.feat_dim, self.proj_dim, 1, 1, 0)
        self.state_init_projection = nn.Conv2d(self.feat_dim, self.lm_state_dim, 1, 1, 0)
        self.lm_rnn1 = nn.GRUCell(input_size=self.feat_dim, hidden_size=self.lm_state_dim)
        self.lm_rnn2 = nn.GRUCell(input_size=self.feat_dim, hidden_size=self.lm_state_dim)
        self.image_attention = ImageAttention(self.proj_dim, self.feat_dim + self.lm_state_dim, cover_kernel)
        self.struct_cls = nn.Sequential(
            nn.Linear(self.feat_dim + self.lm_state_dim, self.lm_state_dim),
            nn.Tanh(),
            nn.Linear(self.lm_state_dim, len(self.vocab))
        )

    def init_state(self, feats, feats_mask):
        bs, _, h, w = feats.shape
        project_feats = self.feat_projection(feats) * feats_mask
        init_state = torch.sum(self.state_init_projection(feats), dim=(2, 3))/torch.sum(feats_mask, dim=(2, 3))
        init_context = torch.sum(feats, dim=(2, 3)) / torch.sum(feats_mask, dim=(2, 3))
        init_spatial_att_weight = torch.zeros([bs, 1, h, w], dtype=torch.float, device=feats.device)
        init_cum_spatial_att_weight = torch.zeros([bs, 1, h, w], dtype=torch.float, device=feats.device)
        return project_feats, init_state, init_context, init_spatial_att_weight, init_cum_spatial_att_weight

    def step(self, feats, project_feats, feats_mask, state, context, spatial_att_weight, cum_spatial_att_weight, layouts=None, layouts_cum=None, spatial_att_weight_scores=None):
        new_state = self.lm_rnn1(context, state)
        new_state, new_context, new_spatial_att_logit, \
            new_spatial_att_weight, new_cum_spatial_att_weight, \
                layouts_cum, spatial_att_weight_scores = self.image_attention(
            project_feats,
            feats_mask,
            torch.cat([context, new_state], dim=1),
            spatial_att_weight,
            cum_spatial_att_weight,
            feats,
            new_state,
            layouts,
            layouts_cum,
            spatial_att_weight_scores
        )
        new_state = self.lm_rnn2(new_context, new_state)
        cls_feat = torch.cat([new_context, new_state], dim=1)
        cls_logits_pt = self.struct_cls(cls_feat)
        return cls_logits_pt, new_state, new_context, new_spatial_att_logit, new_spatial_att_weight, new_cum_spatial_att_weight, layouts_cum, spatial_att_weight_scores

    def forward(self, feats, feats_mask, cls_labels=None, labels_mask=None, layouts=None):
        if self.training:
            return self.forward_backward(feats, feats_mask, cls_labels, labels_mask, layouts)
        else:
            return self.inference(feats, feats_mask)
    
    def inference(self, feats, feats_mask):
        bs, _, h, w = feats.shape
        device = feats.device
        assert bs == 1, print('bs should be 1')
        layouts_cum = torch.zeros_like(feats[:, : 1])
        spatial_att_weight_scores = torch.zeros(bs).to(device=device, dtype=feats.dtype)

        project_feats, init_state, init_context, spatial_att_weight, cum_spatial_att_weight = self.init_state(feats, feats_mask)
        state = init_state
        context = init_context

        for _ in range(h*w):
            cls_logits_pt, state, context, spatial_att_logit, spatial_att_weight, \
                cum_spatial_att_weight, layouts_cum, spatial_att_weight_scores \
                    = self.step(
                            feats, project_feats,
                            feats_mask, state, context,
                            spatial_att_weight, cum_spatial_att_weight, None, layouts_cum, spatial_att_weight_scores)
            feats = feats[:1].repeat(layouts_cum.shape[0], 1, 1, 1)
            feats_mask = feats_mask[:1].repeat(layouts_cum.shape[0], 1, 1, 1)
            project_feats = project_feats[:1].repeat(layouts_cum.shape[0], 1, 1, 1)
            if cum_spatial_att_weight.min() == 1:
                break
        spatial_att_logit_preds = layouts_cum[spatial_att_weight_scores.argmax(), 1:].unsqueeze(0)
        return spatial_att_logit_preds, {}

    def forward_backward(self, feats, feats_mask, cls_labels, labels_mask, layouts):
        device = feats.device
        valid_cls_length = torch.sum((labels_mask == 1) & (cls_labels != -1), dim=1).detach()
        valid_spatial_att_logit_length = torch.stack([layout.max() + 1 for layout in layouts])
        max_length = valid_cls_length.max()
       
        project_feats, init_state, init_context, spatial_att_weight, cum_spatial_att_weight = self.init_state(feats, feats_mask)
        state = init_state
        context = init_context
        
        loss_cache = dict()

        cls_loss = list()
        cls_preds = list()

        spatial_att_logit_loss = list()
        spatial_att_logit_preds = list()
        spatial_att_logit_masks = list()
        spatial_att_logit_labels = list()
        for time_t in range(max_length):
            cls_logits_pt, state, context, spatial_att_logit, spatial_att_weight, cum_spatial_att_weight, *_  \
                = self.step(
                        feats, project_feats,
                        feats_mask, state, context,
                        spatial_att_weight, cum_spatial_att_weight, layouts == time_t
                    )

            cls_label = cls_labels[:, time_t]
            label_mask = labels_mask[:, time_t]
            # cal cls loss
            cls_loss_pt = F.cross_entropy(cls_logits_pt, cls_label, ignore_index=-1, reduction='none') * label_mask 
            cls_loss.append(cls_loss_pt)
            # save for acc
            cls_preds.append(torch.argmax(cls_logits_pt, dim=1).detach())
            
            spatial_att_logit_preds.append(spatial_att_logit.sigmoid() > self.att_threshold)
            spatial_att_logit_masks.append((layouts != -1).unsqueeze(1))
            spatial_att_logit_labels.append((layouts == time_t).unsqueeze(1))
            # cal spatial att loss
            spatial_att_logit_loss_pt = list()
            for spatial_att_logit_pi, layout in zip(spatial_att_logit, layouts):
                target = layout == time_t
                if torch.any(target) == False:
                    spatial_att_logit_loss_pt_pi = torch.tensor(0.0, dtype=torch.float, device=device)
                else:
                    mask = (layout != -1).float()
                    spatial_att_logit_loss_pt_pi = F.binary_cross_entropy_with_logits(
                        spatial_att_logit_pi,
                        target.float().unsqueeze(0),
                        reduction='none'
                    )
                    spatial_att_logit_loss_pt_pi = (spatial_att_logit_loss_pt_pi * mask).sum()
                spatial_att_logit_loss_pt.append(spatial_att_logit_loss_pt_pi)
            spatial_att_logit_loss_pt = torch.stack(spatial_att_logit_loss_pt, dim=0)
            spatial_att_logit_loss.append(spatial_att_logit_loss_pt)
        
        cls_loss = torch.mean(torch.sum(torch.stack(cls_loss, dim=1), dim=1)/valid_cls_length)
        spatial_att_logit_loss = self.spatial_att_logit_loss_wight * torch.mean(torch.sum(torch.stack(spatial_att_logit_loss, dim=1), dim=1) / valid_spatial_att_logit_length)

        loss_cache['cls_loss'] = cls_loss
        loss_cache['spatial_att_logit_loss'] = spatial_att_logit_loss

        cls_preds = torch.stack(cls_preds, dim=1)
        spatial_att_logit_preds = torch.stack(spatial_att_logit_preds, dim=1)
        spatial_att_logit_masks = torch.stack(spatial_att_logit_masks, dim=1)
        spatial_att_logit_labels = torch.stack(spatial_att_logit_labels, dim=1)
        
        acc_metric = AccMetric()
        cell_merge_acc = CellMergeAcc()
        cls_correct, cls_total = acc_metric(cls_preds, cls_labels, labels_mask)
        cls_none_correct, cls_none_total = acc_metric(cls_preds, cls_labels, (labels_mask == 1) & (cls_labels == self.vocab.none_id))
        cls_bold_correct, cls_bold_total = acc_metric(cls_preds, cls_labels, (labels_mask == 1) & (cls_labels == self.vocab.bold_id))
        cls_space_correct, cls_space_total = acc_metric(cls_preds, cls_labels, (labels_mask == 1) & (cls_labels == self.vocab.space_id))
        cls_blank_correct = cls_none_correct + cls_bold_correct + cls_space_correct
        cls_blank_total = cls_none_total + cls_bold_total + cls_space_total
        cells_correct_nums, cells_total_nums = cell_merge_acc(spatial_att_logit_preds, spatial_att_logit_labels, spatial_att_logit_masks)
        loss_cache['cls_acc'] = cls_correct / cls_total
        loss_cache['cls_none_acc'] = cls_none_correct / cls_none_total
        loss_cache['cls_bold_acc'] = cls_bold_correct / cls_bold_total
        loss_cache['cls_space_acc'] = cls_space_correct / cls_space_total
        loss_cache['cls_blank_acc'] = cls_blank_correct / cls_blank_total
        loss_cache['spatial_att_logit_acc'] = cells_correct_nums / cells_total_nums

        return (spatial_att_logit_preds), loss_cache

def build_decoder(cfg):
    decoder = Decoder(
        vocab=cfg.vocab,
        feat_dim=cfg.encode_dim,
        line_dim=cfg.extractor_dim,
        embed_dim=cfg.embed_dim,
        lm_state_dim=cfg.lm_state_dim,
        proj_dim=cfg.proj_dim,
        hidden_dim=cfg.hidden_dim,
        cover_kernel=cfg.cover_kernel,
        max_length=cfg.max_length
    )
    return decoder

