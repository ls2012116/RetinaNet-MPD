import torch
import torch.nn as nn
import torchvision.models as models
import math
import cv2
import os
import shutil
import numpy as np

from models.fpn import FPN, LastLevelP6P7, Feature_Enhance
from models.heads import CLSHead, REGHead  # , MultiHead
from models.anchors import Anchors
from models.losses import IntegratedLoss  # , KLLoss
from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes, rbox_2_quad
from utils.utils import hyp_parse
from torch import nn
from torch.nn.parameter import Parameter
from models.resnet50 import resnet50
import torch.nn.functional as F
from functools import partial
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

nonlinearity = partial(F.relu, inplace=True)


class MultiScaleFusion(nn.Module):

    def __init__(self) -> None:
        super(MultiScaleFusion, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(256 * 3, 256, 3, 1, 1)

    def forward(self, x1, x2, x3):
        x1 = self.max_pool(x1)
        x3 = F.upsample(x3, size=x2.size()[2:], mode='bilinear')
        # x1: torch.Size([8, 256, 26, 40])
        # x2: torch.Size([8, 256, 26, 40])
        # x3: torch.Size([8, 256, 26, 40])
        # torch.Size([8, 256, 52, 80])

        fina_fuse = torch.cat((x1, x2, x3), 1)

        fina_fuse = self.conv(fina_fuse)

        # print('final_fuse: ',fina_fuse.shape)
        return fina_fuse


class ChannelGate(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(ChannelGate, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class SpatialGate(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(SpatialGate, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class RetinaNet(nn.Module):
    def __init__(self, backbone='res50', hyps=None):
        super(RetinaNet, self).__init__()
        self.num_classes = int(hyps['num_classes']) + 1
        self.anchor_generator = Anchors(
            ratios=np.array([0.5, 1, 2]),
        )
        self.num_anchors = self.anchor_generator.num_anchors
        self.init_backbone(backbone)
        # self.dblock = Dblock()
        self.fpn = FPN(
            in_channels_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256),
            use_asff=False
        )
        self.muti_scale_fusion = MultiScaleFusion()
        self.cls_attention = Feature_Enhance(mode='cls')
        self.reg_attention = Feature_Enhance(mode='reg')
        # self.cls_attention = SpatialGate(256)
        #
        # self.reg_attention = ChannelGate(256)

        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes
        )
        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_regress=5  # xywha
        )
        self.loss = IntegratedLoss(func='smooth')
        # self.loss_var = KLLoss()
        self.box_coder = BoxCoder()

    def init_backbone(self, backbone):
        if backbone == 'res34':
            self.backbone = models.resnet34(pretrained=True)
            self.fpn_in_channels = [128, 256, 512]
        elif backbone == 'res50':
            self.backbone = models.resnext50_32x4d(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        elif backbone == 'res101':
            self.backbone = models.resnext101_32x8d(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        elif backbone == 'res152':
            self.backbone = models.resnet152(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        elif backbone == 'resnext50':
            self.backbone = models.resnext50_32x4d(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        else:
            raise NotImplementedError
        del self.backbone.avgpool
        del self.backbone.fc

    def ims_2_features(self, ims):
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(ims)))
        c2 = self.backbone.layer1(self.backbone.maxpool(c1))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)

        # c_i shape: bs,C,H,W
        return [c1, c2, c3, c4, c5, ]

    def forward(self, ims, gt_boxes=None, test_conf=None, process=None, ):

        anchors_list, offsets_list, cls_list, var_list = [], [], [], []
        original_anchors = self.anchor_generator(ims)  # (bs, num_all_achors, 5)
        anchors_list.append(original_anchors)
        features = self.fpn(self.ims_2_features(ims))
        # fina_fuse=torch.cat((features[0],features[1], features[2]))
        features = list(features)
        # features[4] = F.upsample(features[4], size=features[3].size()[2:], mode='bilinear')
        # print('x0',features[0].shape)
        # print('x1', features[1].shape)
        # print('x2', features[2].shape)
        # print('x3', features[3].shape)
        # print('x4', features[4].shape)
        # MultiScaleFusion = self.muti_scale_fusion(features[0], features[1], features[2])
        MultiScaleFusion1 = self.muti_scale_fusion(features[1], features[2], features[3])
        features[2] = MultiScaleFusion1
        cls_feats = self.cls_attention(features, process)
        reg_feats = self.reg_attention(features, process)

        cls_score = torch.cat([(self.cls_head(feature)) for feature in cls_feats], dim=1)
        bbox_pred = torch.cat([(self.reg_head(feature)) for feature in reg_feats], dim=1)
        bboxes = self.box_coder.decode(anchors_list[-1], bbox_pred, mode='xywht').detach()

        if self.training:
            losses = dict()
            bf_weight = self.calc_mining_param(process, 0.3)
            losses['loss_cls'], losses['loss_reg'] = self.loss(cls_score, bbox_pred, anchors_list[-1], bboxes, gt_boxes, \
                                                               md_thres=0.6,
                                                               mining_param=(bf_weight, 1 - bf_weight, 5)
                                                               )
            return losses

        else:  # eval() mode
            return self.decoder(ims, anchors_list[-1], cls_score, bbox_pred, test_conf=test_conf)

    def decoder(self, ims, anchors, cls_score, bbox_pred, thresh=0.6, nms_thresh=0.2, test_conf=None):
        if test_conf is not None:
            thresh = test_conf
        bboxes = self.box_coder.decode(anchors, bbox_pred, mode='xywht')
        bboxes = clip_boxes(bboxes, ims)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]
        # NMS
        anchors_nms_idx = nms(torch.cat([bboxes, scores], dim=2)[0, :, :], nms_thresh)
        nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            dim=1
        )
        return [nms_scores, nms_class, output_boxes]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def calc_mining_param(self, process, alpha):
        if process < 0.1:
            bf_weight = 1.0
        elif process > 0.3:
            bf_weight = alpha
        else:
            bf_weight = 5 * (alpha - 1) * process + 1.5 - 0.5 * alpha
        return bf_weight


if __name__ == '__main__':
    pass
