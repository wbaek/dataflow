import os
import copy

import ujson as json
import requests
import numpy as np
import cv2

import tensorpack.dataflow as df
from tensorpack.dataflow.base import RNGDataFlow

import logging
logger = logging.getLogger(__name__)

class NetworkImages(RNGDataFlow):
    def __init__(self, shuffle=False):
        # will be implement datapoints
        #self.datapoints = [['http://image_url', labels...], ]
        self.shuffle = shuffle
        self.is_parallel = False
        
    def size(self):
        return len(self.datapoints)

    @staticmethod
    def read(url):
        for trial in range(5):
            try:
                resp = requests.get(url)
                if resp.status_code // 100 != 2:
                     logger.warning('request failed code=%d url=%s' % (resp.status_code, url))
                     time.sleep(0.05)
                     continue
                return resp.content
            except Exception as e:
                logger.warning('request failed error=%s url=%s' % (str(e), url))
        return None

    @staticmethod
    def map_func_download(datapoint):
        content = NetworkImages.read(datapoint[0])
        return [content] + datapoint[1:]

    @staticmethod
    def map_func_decode(datapoint):        
        img = cv2.imdecode(np.fromstring(datapoint[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        return [img] + datapoint[1:]

    def get_data(self):
        idxs = np.arange(len(self.datapoints))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for i, k in enumerate(idxs):
            dp = self.datapoints[k]
            if not self.is_parallel:
                dp = NetworkImages.map_func_decode( NetworkImages.map_func_download( dp ) )
            yield dp

    def partitioning(self, num_partition, partition_index=0):
        if num_partition <= partition_index:
            raise ValueError('partition_index(=%d) is must be a smaller than num_partition(=%d)'%(partition_index, num_partition)) 
        self.datapoints = self.datapoints[partition_index::num_partition]
        return self

    def parallel(self, num_threads, buffer_size=200, strict=False):
        self.is_parallel = True
        ds = self
        ds = df.MultiThreadMapData(ds,
                                   nr_thread=num_threads, map_func=NetworkImages.map_func_download,
                                   buffer_size=buffer_size, strict=strict)
        ds = df.PrefetchDataZMQ(ds, nr_proc=1) # to reduce GIL contention.
        ds = df.MapData(ds, func=NetworkImages.map_func_decode)
        return ds

