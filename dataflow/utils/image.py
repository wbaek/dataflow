import random

import cv2
from tensorpack.dataflow.base import ProxyDataFlow

import logging
logger = logging.getLogger(__name__)


class Viewer(ProxyDataFlow):
    def __init__(self, ds, condition=lambda x: x[1] is 0, name='label-0', prob=0.1, pos=(0, 0)):
        super(Viewer, self).__init__(ds)
        self.condition = condition
        self.name = name
        self.prob = prob
        self.pos = pos

    def get_data(self):
        for dp in self.ds.get_data():
            if self.condition(dp) and random.random() <= self.prob:
                cv2.namedWindow(self.name)
                cv2.moveWindow(self.name, self.pos[0], self.pos[1])

                cv2.imshow(self.name, dp[0])
                cv2.waitKey(1)

            if dp is not None:
                yield dp
