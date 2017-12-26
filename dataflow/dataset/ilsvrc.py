from .base import NetworkImages

import os
import ujson as json
import requests

import logging
logger = logging.getLogger(__name__)


class ILSVRC12(NetworkImages):
    def __init__(self, service_code, train_or_valid, shuffle=True):
        super(ILSVRC12, self).__init__(shuffle)

        base_path = 'http://twg.kakaocdn.net/%s/imagenet/ILSVRC/2012/object_localization/ILSVRC/' % service_code
        data_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../datas/ILSVRC/classification')
        temp_map = json.load(open(data_path+'/imagenet1000_classid_to_text_synsetid.json'))
        self.maps = {
            'idx2synset': {int(key): value['id'] for key, value in iter(temp_map.items())},
            'synset2idx': {value['id']: int(key) for key, value in iter(temp_map.items())},
            'idx2text':   {int(key): value['text'] for key, value in iter(temp_map.items())}
        }
        if train_or_valid in ['train', 'training']:
            _ = [line.decode('utf-8').split(' ')[0]
                 for line in
                 requests.get(base_path + 'ImageSets/CLS-LOC/train_cls.txt').content.splitlines() if line]
            self.datapoints = [
                [base_path + 'Data/CLS-LOC/train/'+line+'.JPEG', int(self.maps['synset2idx'][line.split('/')[0]])]
                for line in
                _
            ]
        elif train_or_valid in ['valid', 'validation']:
            synsets = [line.strip()
                       for line in
                       open(data_path+'/imagenet_2012_validation_synset_labels.txt').readlines()]
            self.datapoints = [
                [base_path + 'Data/CLS-LOC/val/ILSVRC2012_val_%08d.JPEG' % (i+1), int(self.maps['synset2idx'][synset])]
                for i, synset in enumerate(synsets)
            ]
        else:
            raise ValueError('train_or_valid=%s is invalid argument must be a set train or valid' % train_or_valid)


if __name__ == '__main__':
    import argparse
    import tensorpack.dataflow as df

    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--service-code', type=str, required=True,
                        help='licence key')
    parser.add_argument('--name', type=str, default='train',
                        help='train or valid')
    args = parser.parse_args()

    ds = ILSVRC12(args.service_code, args.name).parallel(num_threads=8)
    if args.name in ['train', 'training']:
        ds = df.PrefetchDataZMQ(ds, nr_proc=2)
    
    df.TestDataSpeed(ds, size=5000).start()
