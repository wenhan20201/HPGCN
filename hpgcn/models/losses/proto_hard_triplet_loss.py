import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseWeightedLoss


class ProtoHardTripletLoss(BaseWeightedLoss):
    def __init__(self, margin=0.3, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.margin = margin

    def _forward(self, proto_feat, label):
        """
        proto_feat: [N, 32]  高层原型特征
        label:      [N]     标签
        """
        N = proto_feat.size(0)
        feat = F.normalize(proto_feat, dim=-1)

        # 距离矩阵
        dist = torch.cdist(feat, feat)  # N, N

        # 掩码
        same_mask = label.unsqueeze(1) == label.unsqueeze(0)  # N,N
        diff_mask = ~same_mask

        # 最难正样本：同类最远
        ap = dist.masked_fill(~same_mask, -1e10).max(dim=1)[0]

        # 最难负样本：异类最近
        an = dist.masked_fill(~diff_mask, 1e10).min(dim=1)[0]

        # 三元损失
        loss = F.relu(ap - an + self.margin).mean()
        return loss