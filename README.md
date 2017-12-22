# dataflow
data feeder using [tensorpack.dataflow](http://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.html)

## Introduce
> Dataflow in Tensorpack
> 1. It's easy: write everything in pure Python.
> 2. It's fast: see Efficient DataFlow on how to build a fast DataFlow with parallelism.
>
> -- http://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.html

## Examples

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

