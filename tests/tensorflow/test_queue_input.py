from dataflow.tensorflow.queue_input import QueueInput

import time
import pytest
import tensorflow as tf
import tensorpack.dataflow as df


def test_testdata():
    ds = df.DataFromList(list(zip(range(10), range(10, 0, -1))), shuffle=False)
    ds.reset_state()

    index = 0
    for dp in ds.get_data():
        assert index == dp[0]
        assert 10-index == dp[1]
        index += 1

def test_queue_input():
    ds = df.DataFromList(list(zip(range(10), range(10, 0, -1))), shuffle=False)
    ds.reset_state()

    placeholders = [tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())]
    thread = QueueInput(ds, placeholders)

    tensors = [2.0*ph for ph in thread.tensors()]

    with tf.Session() as sess:
        thread.start()

        for i in range(10):
            dp = sess.run(tensors)
            assert i*2.0 == dp[0]
            assert (10-i)*2.0 == dp[1]

def test_queue_input_terminated():
    ds = df.DataFromList(list(zip(range(10), range(10, 0, -1))), shuffle=False)
    ds.reset_state()

    placeholders = [tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())]
    thread = QueueInput(ds, placeholders)

    tensors = [2.0*ph for ph in thread.tensors()]

    with tf.Session() as sess:
        thread.start()

        for i in range(10):
            dp = sess.run(tensors)

        with pytest.raises(tf.errors.OutOfRangeError) as excinfo:
            sess.run(tensors)

def test_queue_input_infinite():
    ds = df.DataFromList(list(zip(range(10), range(10, 0, -1))), shuffle=False)
    ds.reset_state()

    placeholders = [tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())]
    thread = QueueInput(ds, placeholders, repeat_infinite=True)

    tensors = [2.0*ph for ph in thread.tensors()]

    with tf.Session() as sess:
        thread.start()

        for i in range(30):
            dp = sess.run(tensors)

def test_queue_input_infinite():
    ds = df.DataFromList(list(zip(range(10), range(10, 0, -1))), shuffle=False)
    ds.reset_state()

    placeholders = [tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())]
    thread = QueueInput(ds, placeholders, repeat_infinite=True, queue_size=50)

    tensors = [2.0*ph for ph in thread.tensors()]

    with tf.Session() as sess:
        assert 0 ==  sess.run(thread.queue_size())
        thread.start()

        assert 0 < sess.run(thread.queue_size())
        time.sleep(1.0)
        assert 50 == sess.run(thread.queue_size())

def test_queue_input_multi_threads():
    threads = []
    for t in range(3): 
        ds = df.DataFromList(list(zip(range(10), range(10, 0, -1))), shuffle=False)
        ds.reset_state()

        placeholders = [tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())]
        thread = QueueInput(ds, placeholders, repeat_infinite=True)
        threads.append(thread)

    with tf.Session() as sess:
        for thread in threads:
            thread.start()

        for i in range(10):
            dp = sess.run([t.tensors() for t in threads])
            assert [[i*1.0, (10-i)*1.0]] * 3


