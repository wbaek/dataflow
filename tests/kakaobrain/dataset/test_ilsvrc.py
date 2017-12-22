from dataflow.kakaobrain.dataset.ilsvrc import ILSVRC12

import numpy as np

def test_ilsvrc12_read_url():
    response = ILSVRC12.read('https://httpbin.org/robots.txt')
    assert response.decode('utf-8') == '''User-agent: *
Disallow: /deny
'''

    response = ILSVRC12.read('https://httpbin.org/status/404')
    assert response == None

def test_ilsvrc12():
    ds = ILSVRC12('braincloud', 'train')

def test_ilsvrc12_get_data():
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

def test_ilsvrc12_get_data_shuffled():
    ds = ILSVRC12('braincloud', 'train', shuffle=True)

    ds.reset_state()
    idx = 0
    for dp in ds.get_data():
        assert dp[1] != 0
        assert dp[0].dtype == np.uint8
        assert np.min(dp[0]) == 0
        assert np.max(dp[0]) == 255
        idx += 1
        if idx > 3:
            break


