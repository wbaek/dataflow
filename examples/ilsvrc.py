import sys
import copy
import logging

import tensorpack.dataflow as df
import dataflow

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--service-code', type=str, required=True,
                        help='licence key')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--log-filename',   type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    augmentors = [
        df.imgaug.Resize((128, 128)),
    ]

    ds = dataflow.dataset.ILSVRC12(args.service_code, 'train', shuffle=True).parallel(num_threads=16)
    ds = df.AugmentImageComponent(ds, augmentors)
    ds = df.PrefetchDataZMQ(ds, nr_proc=2)
    if args.view:
        ds = dataflow.utils.image.Viewer(ds, lambda x: x[1] == 4,  'label-4',  prob=1.0, pos=(0, (128+64)*0))
        ds = dataflow.utils.image.Viewer(ds, lambda x: x[1] == 16, 'label-16', prob=1.0, pos=(0, (128+64)*1))
        ds = dataflow.utils.image.Viewer(ds, lambda x: x[1] == 32, 'label-32', prob=1.0, pos=(0, (128+64)*2))

    ds.reset_state()
    df.TestDataSpeed(ds, size=5000).start()
