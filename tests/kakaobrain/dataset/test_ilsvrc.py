from dataflow.kakaobrain.dataset.ilsvrc import ILSVRC12

import numpy as np
import tensorpack.dataflow as df

def test_ilsvrc12_read_url():
    response = ILSVRC12.read('https://httpbin.org/robots.txt')
    assert response.decode('utf-8') == '''User-agent: *
Disallow: /deny
'''

    response = ILSVRC12.read('https://httpbin.org/status/404')
    assert response == None

def test_ilsvrc12_map_func_download():
    dp = ILSVRC12.map_func_download(['https://httpbin.org/robots.txt', 'kakao', 'brain'])
    assert dp[0].decode('utf-8') == '''User-agent: *
Disallow: /deny
'''
    assert dp[1] == 'kakao'
    assert dp[2] == 'brain'

def test_ilsvrc12_map_func_decode():
    dp = ILSVRC12.map_func_download(['https://httpbin.org/image/jpeg', 'kakao', 'brain'])
    dp = ILSVRC12.map_func_decode(dp)
    assert dp[0].shape == (178, 239, 3)
    assert dp[1] == 'kakao'
    assert dp[2] == 'brain'

def test_ilsvrc12():
    ds = ILSVRC12('braincloud', 'train')

def test_ilsvrc12_get_data():
    ds = ILSVRC12('braincloud', 'train', shuffle=False)
    ds = df.MultiThreadMapData(ds, nr_thread=1, map_func=ILSVRC12.map_func_download)
    ds = df.MapData(ds, func=ILSVRC12.map_func_decode)
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
    ds = df.MultiThreadMapData(ds, nr_thread=1, map_func=ILSVRC12.map_func_download)
    ds = df.MapData(ds, func=ILSVRC12.map_func_decode)

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


