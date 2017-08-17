# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:18:45 2017

@author: yamane
"""

import chainer
import chainer.functions as F
import chainer.links as L

class CBR(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(CBR, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize=None,
                                 stride=1, pad=0, nobias=False, initialW=None,
                                 initial_bias=None, **kwargs),
            bn=L.BatchNormalization(out_channels)
        )

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))

class CndBR(chainer.Chain):
    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False):
        super(CndBR, self).__init__(
            conv=L.ConvolutionND(ndim, in_channels, out_channels, ksize,
                                 stride=1, pad=0, nobias=False, initialW=None,
                                 initial_bias=None, cover_all=False),
            bn=L.BatchNormalization(out_channels)
        )

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))

#class SpatialConv(chainer.Chain):
#    def __init__(self):
#        super(SpatialConv, self).__init__(
#            cbr1=CBR(4, 32, ksize=3, pad=1),
#            cbr2= CBR(32, 64, ksize=3, pad=1),
#            cbr3= CBR(64, 128, ksize=3, pad=1),
#            fc1=L.Linear(None, 256),
#            fc2=L.Linear(256, 18)
#        )
#
#    def __call__(self, x):
#        h = self.cbr1(x)
#        h = F.max_pooling_2d(h, 3)
#        h = self.cbr2(h)
#        h = F.max_pooling_2d(h, 3)
#        h = self.cbr3(h)
#        h = F.max_pooling_2d(h, 3)
#        h = F.relu(self.fc4(h))
#        h = F.dropout(h)
#        y = self.fc5(h)
#        return y
#
#class SpatialConv2(chainer.ChainList):
#    def __init__(self):
#        super(SpatialConv, self).__init__(
#            CBR(4, 32, ksize=3, pad=1),
#            CBR(32, 64, ksize=3, pad=1),
#            CBR(64, 128, ksize=3, pad=1),
#            L.Linear(None, 256),
#            L.Linear(256, 18)
#            )
#    def __call__(self, x):
#        h = x
#        for link in self[:-2]:
#            h = link(h)
#            h = F.max_pooling_2d(h, 3)
#        h = F.relu(self[-2](h))
#        h = F.dropout(h)
#        y = self[-1](h)
#        return y
