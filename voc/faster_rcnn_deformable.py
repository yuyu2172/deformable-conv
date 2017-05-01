import numpy as np

import chainer
import chainer.functions as F
from chainer.initializers import constant
import chainer.links as L
from chainer.links.model.vision.resnet import ResNet101Layers
from chainer.links.model.vision.resnet import BuildingBlock
from chainer.links.model.vision.resnet import BottleneckA
from chainer.links.model.vision.resnet import BottleneckB
from chainer.links.model.vision.resnet import relu
from chainer.links.connection.deformable_convolution_2d import DeformableConvolution2D

from chainercv.links.faster_rcnn.rpn import RPN
from chainercv.links.faster_rcnn.faster_rcnn import FasterRCNNBase

from chainer import link
from chainer.links import Convolution2D
from chainer.links import BatchNormalization


class DeformableBottleneckB(link.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(DeformableBottleneckB, self).__init__(
            conv1=Convolution2D(
                in_channels, mid_channels, 1, 1, 0,
                initialW=initialW, nobias=True),
            bn1=BatchNormalization(mid_channels),
            conv2=DeformableConvolution2D(
                mid_channels, mid_channels, 3, 1, 1,
                initialW=initialW, nobias=True),
            bn2=BatchNormalization(mid_channels),
            conv3=Convolution2D(
                mid_channels, in_channels, 1, 1, 0,
                initialW=initialW, nobias=True),
            bn3=BatchNormalization(in_channels),
        )

    def __call__(self, x, test=True):
        h = relu(self.bn1(self.conv1(x), test=test))
        h = relu(self.bn2(self.conv2(h), test=test))
        h = self.bn3(self.conv3(h), test=test)
        return relu(h + x)


class BuildingBlockDeformable(link.Chain):

    """A building block that consists of several Bottleneck layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):

        links = [
            ('a', BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW))
        ]
        links.append(('b1', BottleneckB(out_channels, mid_channels, initialW)))
        links.append(('b2', BottleneckB(out_channels, mid_channels, initialW)))
        super(BuildingBlockDeformable, self).__init__(**dict(links))
        self.forward = links

    def __call__(self, x, test=True):
        for name, func in self.forward:
            x = func(x, test=test)
        return x


class FasterRCNNHeadResNetDeformable(chainer.Chain):

    def __init__(self, n_class, initialW=None):
        cls_init = chainer.initializers.Normal(0.01)
        bbox_init = chainer.initializers.Normal(0.001)
        super(FasterRCNNHeadResNetDeformable, self).__init__(
            res5=BuildingBlockDeformable(
                3, 1024, 512, 2048, 2, initialW=initialW),
            cls_score=L.Linear(2048, n_class, initialW=cls_init),
            bbox_pred=L.Linear(2048, n_class * 4, initialW=bbox_init),
        )

    def __call__(self, x, train=False):
        h = self.res5(x, test=not train)
        h = F.max_pooling_2d(h, ksize=7)

        cls_score = self.cls_score(h)
        bbox_pred = self.bbox_pred(h)
        return bbox_pred, cls_score


class FasterRCNNResNetDeformable(FasterRCNNBase):

    def __init__(self, n_class=21,
                 nms_thresh=0.3, conf_thresh=0.05,
                 n_anchors=9, anchor_scales=[8, 16, 32],
                 targets_precomputed=True
                 ):
        feat_stride = 16
        rpn_sigma = 3.
        sigma = 1.

        feature = ResNet101Layers()
        rpn = RPN(1024, 256, n_anchors, feat_stride,
                  anchor_scales, n_class, rpn_sigma=rpn_sigma)
        head = FasterRCNNHeadResNetDeformable(n_class, initialW=constant.Zero())

        super(FasterRCNNResNetDeformable, self).__init__(
            feature,
            rpn,
            head,
            n_class=n_class,
            roi_size=14,
            nms_thresh=nms_thresh,
            conf_thresh=conf_thresh,
            sigma=sigma,
        )
        # Handle pretrained models
        self.head.res5.a.copyparams(self.feature.res5.a)
        # def copy_res5_to_deform_res5(target, source):
        #     target.conv1.copyparams(source.conv1)
        #     target.bn1.copyparams(source.bn1)
        #     target.conv2.deform_conv.copyparams(source.conv2)
        #     target.bn2.copyparams(source.bn2)
        #     target.conv3.copyparams(source.conv3)
        #     target.bn3.copyparams(source.bn3)
        # copy_res5_to_deform_res5(self.head.res5.b1, self.feature.res5.b1)
        # copy_res5_to_deform_res5(self.head.res5.b2, self.feature.res5.b2)

        remove_links = ['res5']
        for name in remove_links:
            self.feature._children.remove(name)
            delattr(self.feature, name)

    def _extract_feature(self, x):
        hs = self.feature(x, layers=['res2', 'res4'])
        h = hs['res4']
        hs['res2'].unchain_backward()
        return h
