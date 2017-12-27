# dataflow
data feeder using [tensorpack.dataflow](http://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.html)

## Introduce
> Dataflow in Tensorpack
> 1. It's easy: write everything in pure Python.
> 2. It's fast: see Efficient DataFlow on how to build a fast DataFlow with parallelism.
>
> -- http://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.html

## Examples
* General network images
```pytho
from dataflow.dataset import NetworkImages

class NetworkImagesImple(NetworkImages):
    def __init__(self, shuffle=False):
        super(NetworkImagesImple, self).__init__(shuffle)
        self.datapoints = [
            ['http://t1.daumcdn.net/news/201511/20/sportskhan/20151120010041631lkva.jpg', 0],
            ['http://t1.daumcdn.net/news/201511/03/SpoChosun/20151103111905902jtmo.jpg',  1],
            ['http://t1.daumcdn.net/news/201712/26/ked/20171226081404015hktd.jpg',        2],
            ['http://t1.daumcdn.net/news/201511/05/10asia/20151105173913995tqqc.jpg',     3],
            ['http://t1.daumcdn.net/news/201607/20/etimesi/20160720112503626xuwr.jpg',    4],
        ]

ds = NetworkImagesImple()

for datapoint in ds.get_data():
    pass
```

* ILSVRC12 **multi threaded** downloading with **multi processed** preprocessing
```python
import tensorpack.dataflow as df
from dataflow.dataset import ILSVRC12

service_code = 'CONTACT_ME'
ds = ILSVRC12(service_code, 'train', shuffle=True).parallel(num_threads=16)
ds = df.PrefetchDataZMQ(ds, nr_proc=8)

for datapoint in ds.get_data():
    pass
```

## Original Dataflow Examples

### Basic Dataflow
* [Mnist Preprocessing](https://github.com/wbaek/dataflow/blob/master/examples/tensorpack/mnist.py)
* [Mnist Preprocessing \w viewer](https://github.com/wbaek/dataflow/blob/master/examples/tensorpack/viewimage.py)

### Distributed Dataflow
* [Preprocessor (sender)](https://github.com/wbaek/dataflow/blob/master/examples/tensorpack/distributed/preprocessor.py)
* [Trainer (receiver)](https://github.com/wbaek/dataflow/blob/master/examples/tensorpack/distributed/trainer.py)

### Tensorflow \w Dataflow
* [Mnist Tensorflow \w Dataflow](https://github.com/wbaek/dataflow/blob/master/examples/tensorflow/mnist.py#L6-L17)

### PyTorch \w Dataflow
* [Mnist PyTorch \w Dataflow](https://github.com/wbaek/dataflow/blob/master/examples/pytorch/mnist.py#L8-L21)


## Install

### pre-requirements
* ubuntu
```
apt install -y libsm6 libxext-dev
```

* mac
```
```

* commons
```
pip install -r requirements.txt
```


## Benchmark

### ILSVRC12

#### parallel download and decode image only
* without image augment (parallel download and decode only)
* resource : 4 GPU, 8 CPU, 48 GB in kakaobrain braincloud

##### unit : duration time (5000 images)
| threads \ process | 1 | 2 | 4 | 8 | 16 | 32 |
|-------------------|-------|-------|-------|-------|-------|-------|
| 1                 | 05:17 | 02:37 | 01:25 | 00:36 | 00:18 | 00:08 |
| 2                 | 02:39 | 01:23 | 00:35 | 00:17 | 00:08 | 00:05 |
| 4                 | 01:10 | 00:35 | 00:17 | 00:08 | 00:06 | 00:05 |
| 8                 | 00:35 | 00:17 | 00:08 | **00:05** | 00:06 | 00:08 |
| 16                | 00:25 | 00:13 | 00:06 | 00:06 | 00:07 | 00:09 |
| 32                | 00:26 | 00:13 | 00:06 | 00:06 | 00:08 | 00:09 |


##### unit : images per sec
| threads \ process | 1 | 2 | 4 | 8 | 16 | 32 |
|-------------------|-------|-------|-------|-------|-------|-------|
| 1                 |   15.76 |   31.74 |   58.66 |  135.97 |  269.16 |  556.83 |
| 2                 |   31.42 |   59.81 |  141.52 |  282.79 |  556.39 |  865.39 |
| 4                 |   71.11 |  140.83 |  283.55 |  575.78 |  820.73 |  861.46 |
| 8                 |  141.12 |  286.69 |  555.56 |  **912.18** |  722.68 |  561.70 |
| 16                |  196.69 |  374.15 |  723.51 |  794.82 |  649.93 |  525.28 |
| 32                |  188.49 |  360.05 |  728.51 |  818.10 |  610.04 |  548.91 |


#### parallel downalod and augment for resnet
* resource : 4 GPU, 8 CPU, 48 GB in kakaobrain braincloud

##### unit : duration time (5000 images)
| threads \ process | 2 | 4 | 8 | 16 |
|-------------------|-------|-------|-------|-------|
| 2                 | 01:11 | 00:33 | 00:16 | 00:10 |
| 4                 | 00:33 | 00:16 | **00:08** | 00:12 |
| 8                 | 00:28 | 00:14 | 00:10 | 00:12 |
| 16                | 00:28 | 00:14 | 00:10 | 00:12 |
| 32                | 00:28 | 00:14 | 00:10 | 00:12 |

##### unit : images per sec
| threads \ process | 2 | 4 | 8 | 16 |
|-------------------|--------|--------|--------|--------|
| 2                 |  70.33 | 147.21 | 294.56 | 495.01 |
| 4                 | 149.60 | 303.54 | **539.99** | 397.04 |
| 8                 | 176.30 | 350.24 | 487.05 | 385.01 |
| 16                | 172.77 | 343.41 | 485.52 | 393.25 |
| 32                | 175.74 | 347.32 | 489.17 | 387.76 |
