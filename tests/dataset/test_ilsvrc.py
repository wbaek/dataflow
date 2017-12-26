from dataflow.dataset.ilsvrc import ILSVRC12

import pytest
import numpy as np


def test_ilsvrc12_train_get_data():
    ds = ILSVRC12('braincloud', 'train', shuffle=False)
    assert ds.size() == 1281167

    ds.reset_state()
    for dp in ds.get_data():
        assert dp[1] == 0
        assert dp[0].dtype == np.uint8
        assert np.min(dp[0]) == 0
        assert np.max(dp[0]) == 255
        assert dp[0].shape == (250, 250, 3)
        break


def test_ilsvrc12_train_get_data_shuffled():
    ds = ILSVRC12('braincloud', 'train', shuffle=True)

    ds.reset_state()
    idx = 0
    for dp in ds.get_data():
        assert dp[1] != 0
        assert dp[0].dtype == np.uint8
        assert np.min(dp[0]) <= 30
        assert np.max(dp[0]) >= 220
        idx += 1
        if idx > 3:
            break


def test_ilsvrc12_valid_get_data():
    ds = ILSVRC12('braincloud', 'valid', shuffle=False)
    assert ds.size() == 50000

    ds.reset_state()
    for dp in ds.get_data():
        assert dp[1] == 65
        assert dp[0].dtype == np.uint8
        assert np.min(dp[0]) == 0
        assert np.max(dp[0]) == 255
        assert dp[0].shape == (375, 500, 3)
        break


def test_ilsvrc12_valid_get_data_shuffled():
    ds = ILSVRC12('braincloud', 'valid', shuffle=True)

    ds.reset_state()
    idx = 0
    for dp in ds.get_data():
        assert dp[1] != 65
        assert dp[0].dtype == np.uint8
        assert np.min(dp[0]) <= 30
        assert np.max(dp[0]) >= 220
        idx += 1
        if idx > 3:
            break


def test_ilsvrc12_invalid_name():
    with pytest.raises(ValueError) as excinfo:
        ILSVRC12('braincloud', 'invalid')
    assert 'train_or_valid=invalid is invalid argument must be a set train or valid' == str(excinfo.value)


def test_ilsvrc12_train_partitioning():
    ds = ILSVRC12('braincloud', 'train', shuffle=False).partitioning(10, 0)
    assert ds.size() == 128117

    ds.reset_state()
    for dp in ds.get_data():
        assert dp[1] == 0
        assert dp[0].dtype == np.uint8
        assert np.min(dp[0]) == 0
        assert np.max(dp[0]) == 255
        assert dp[0].shape == (250, 250, 3)
        break

    ds = ILSVRC12('braincloud', 'train', shuffle=False).partitioning(10, 1)
    assert ds.size() == 128117

    ds.reset_state()
    for dp in ds.get_data():
        assert dp[1] == 0
        assert dp[0].dtype == np.uint8
        assert np.min(dp[0]) == 0
        assert np.max(dp[0]) == 255
        assert dp[0].shape == (150, 200, 3)
        break

    ds = ILSVRC12('braincloud', 'train', shuffle=False).partitioning(2000, 0)
    assert ds.size() == 641

    ds.reset_state()
    generator = ds.get_data()
    dp = next(generator)
    assert dp[1] == 0

    dp = next(generator)
    assert dp[1] == 1

    dp = next(generator)
    assert dp[1] == 3


def test_ilsvrc12_train_parallel():
    ds = ILSVRC12('braincloud', 'train', shuffle=False).parallel(num_threads=10, buffer_size=10)
    assert ds.size() == 1281167

    ds.reset_state()
    index = 0
    for dp in ds.get_data():
        assert dp[1] == 0
        if index >= 10:
            break
        index += 1
