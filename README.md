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

* ILSVRC12 multi threaded downloading with multi processed preprocessing
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

