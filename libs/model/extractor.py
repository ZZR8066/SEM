import torch
from torch import nn
from torchvision.ops import roi_align


def convert_to_roi_format(lines_box):
    concat_boxes = torch.cat(lines_box, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full((lines_box_pi.shape[0], 1), i, dtype=dtype, device=device)
            for i, lines_box_pi in enumerate(lines_box)
        ],
        dim=0
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


class RoiFeatExtraxtor(nn.Module):
    def __init__(self, scale, pool_size, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.pool_size = pool_size
        self.output_dim = output_dim
        input_dim = input_dim * self.pool_size[0] * self.pool_size[1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def forward(self, feats, lines_box):
        rois = convert_to_roi_format(lines_box)
        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=self.pool_size,
            spatial_scale=self.scale,
            sampling_ratio=2
        )

        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        lines_feat = self.fc(lines_feat)
        lines_feat = torch.split(lines_feat, [item.shape[0] for item in lines_box])
        return list(lines_feat)


class RoiPosFeatExtraxtor(nn.Module):
    def __init__(self, scale, pool_size, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.pool_size = pool_size
        self.output_dim = output_dim
        input_dim = input_dim * self.pool_size[0] * self.pool_size[1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.bbox_ln = nn.LayerNorm(self.output_dim)
        self.bbox_tranform = nn.Linear(4, self.output_dim)

        self.add_ln = nn.LayerNorm(self.output_dim)

    def forward(self, feats, lines_box, img_sizes):
        rois = convert_to_roi_format(lines_box)
        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=self.pool_size,
            spatial_scale=self.scale,
            sampling_ratio=2
        )
        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        lines_feat = self.fc(lines_feat)
        lines_feat = list(torch.split(lines_feat, [item.shape[0] for item in lines_box]))
        
        # Add Pos Embedding
        feats_H, feats_W = feats.shape[-2:]
        for idx, (line_box, img_size) in enumerate(zip(lines_box, img_sizes)):
            line_box[:, 0] = line_box[:, 0] * self.scale / feats_W
            line_box[:, 1] = line_box[:, 1] * self.scale / feats_H
            line_box[:, 2] = line_box[:, 2] * self.scale / feats_W
            line_box[:, 3] = line_box[:, 3] * self.scale / feats_H
            lines_feat[idx] = self.add_ln(lines_feat[idx] + self.bbox_ln(self.bbox_tranform(line_box)))
        
        return list(lines_feat)
