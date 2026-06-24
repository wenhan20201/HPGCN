import copy as cp
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint
from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import unit_gcn, mstcn, unit_tcn

EPS = 1e-4


class GCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        x, gcl_graph = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), gcl_graph


"""
****************************************
*** 层次化 + 类别专属 原型网络
****************************************
"""
class HierarchicalClassSpecificPrototype(nn.Module):
    def __init__(self, dim=384, num_proto_low=256, num_proto_high=128, num_classes=99, dropout=0.1):
        super().__init__()
        self.proto_low = nn.Linear(dim, num_proto_low, bias=False)
        self.proto_high = nn.Linear(num_proto_low, num_proto_high, bias=False)
        self.class_proto = nn.Embedding(num_classes, num_proto_high)

        self.recover_high = nn.Linear(num_proto_high, num_proto_low)
        self.recover_low = nn.Linear(num_proto_low, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, class_idx=None):
        q_low = self.softmax(self.proto_low(x))
        q_low = self.dropout(q_low)

        q_high = self.softmax(self.proto_high(q_low))
        q_high = self.dropout(q_high)

        if class_idx is not None:
            c_proto = self.class_proto(class_idx).unsqueeze(0)
            q_high = q_high * c_proto

        z_high = self.recover_high(q_high)
        z = self.recover_low(z_high + q_low)
        proto_feat = q_high.mean(0)
        return self.dropout(z), proto_feat


@BACKBONES.register_module()
class HPGCN(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 num_classes=99,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type

        #
        num_proto_low = kwargs.pop('num_proto_low', 64)
        num_proto_high = kwargs.pop('num_proto_high', 32)

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [GCN_Block(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(GCN_Block(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i-1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

        out_channels = base_channels
        norm_cfg = dict(type='BN')
        self.post = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU()

        #  传入层次化参数
        dim = 384
        self.proto_net = HierarchicalClassSpecificPrototype(
            dim=dim,
            num_proto_low=num_proto_low,
            num_proto_high=num_proto_high,
            num_classes=num_classes
        )

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self.pretrained, strict=False)

    def forward(self, x, label=None):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        get_graph = []
        for i in range(self.num_stages):
            x, gcl_graph = self.gcn[i](x)
            get_graph.append(gcl_graph)

        x = x.reshape((N, M) + x.shape[1:])
        c_graph = x.size(2)

        graph = get_graph[-1]
        graph = graph.view(N, M, c_graph, V, V).mean(1).view(N, c_graph, V * V)

        the_graph_list = []
        proto_feat_list = []
        for i in range(N):
            the_graph = graph[i].permute(1, 0)
            if label is not None:
                the_graph, pf = self.proto_net(the_graph, class_idx=label[i:i+1])
            else:
                the_graph, pf = self.proto_net(the_graph)
            the_graph = the_graph.permute(1, 0).view(c_graph, V, V)
            the_graph_list.append(the_graph)
            proto_feat_list.append(pf)

        re_graph = torch.stack(the_graph_list)
        proto_feat = torch.stack(proto_feat_list)
    

        return x, proto_feat