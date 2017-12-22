import os
import ujson as json
import requests
import numpy as np
import cv2

import tensorpack.dataflow as df
from tensorpack.dataflow.base import RNGDataFlow

import logging
logger = logging.getLogger(__name__)

class ILSVRC12(RNGDataFlow):
    def __init__(self, service_code, train_or_test, shuffle=True):
        self.base_path = 'http://twg.kakaocdn.net/{}/imagenet/ILSVRC/2012/object_localization/ILSVRC/'.format(service_code)
        data_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../../datas/ILSVRC/classification')
        temp_map = json.load(open(data_path+'/imagenet1000_classid_to_text_synsetid.json'))
        self.maps = {
            'idx2synset': {int(key):value['id'] for key, value in iter(temp_map.items())},
            'synset2idx': {value['id']:int(key) for key, value in iter(temp_map.items())},
            'idx2text':   {int(key):value['text'] for key, value in iter(temp_map.items())}
        }
        if train_or_test == 'train':
            pass
        else:
            raise NotImplementedError('currently support only train')

        _ = [line.decode('utf-8').split(' ')[0]
            for line in requests.get(self.base_path + 'ImageSets/CLS-LOC/train_cls.txt').content.splitlines() if line]
        self.datapoints = [['Data/CLS-LOC/train/'+line+'.JPEG', self.maps['synset2idx'][line.split('/')[0]]]
            for line in _]
        self.shuffle = shuffle
        
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
        content = ILSVRC12.read(datapoint[0])
        return [content] + datapoint[1:]

    @staticmethod
    def func_decode(img_binary):
        cv2.imdecode(np.fromstring(img_binary, dtype=np.uint8), cv2.IMREAD_COLOR)

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
            url = self.base_path + dp[0]
            yield [url] + dp[1:]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--service-code', type=str, required=True,
                        help='licence key')
    args = parser.parse_args()

    ds = ILSVRC12(args.service_code, 'train')
    ds = df.PrefetchData(ds, 5000, 1)
    ds = df.MultiThreadMapData(ds, nr_thread=16, map_func=ILSVRC12.map_func_download)
    ds = df.MapData(ds, func=ILSVRC12.map_func_decode)
    ds = df.PrefetchDataZMQ(ds, nr_proc=2)
    
    df.TestDataSpeed(ds, size=5000).start()
