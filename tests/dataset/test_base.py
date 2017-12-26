from dataflow.dataset.base import NetworkImages

import pytest
import numpy as np
import tensorpack.dataflow as df


def test_networ_images_read_url():
    response = NetworkImages.read('https://httpbin.org/robots.txt')
    assert response.decode('utf-8') == '''User-agent: *
Disallow: /deny
'''

    response = NetworkImages.read('https://httpbin.org/status/404')
    assert response == None

def test_network_images_map_func_download():
    dp = NetworkImages.map_func_download(['https://httpbin.org/robots.txt', 'kakao', 'brain'])
    assert dp[0].decode('utf-8') == '''User-agent: *
Disallow: /deny
'''
    assert dp[1] == 'kakao'
    assert dp[2] == 'brain'

def test_network_images_map_func_decode():
    dp = NetworkImages.map_func_download(['https://httpbin.org/image/jpeg', 'kakao', 'brain'])
    dp = NetworkImages.map_func_decode(dp)
    assert dp[0].shape == (178, 239, 3)
    assert dp[1] == 'kakao'
    assert dp[2] == 'brain'


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

    @staticmethod
    def test(dp):
        shapes = [(400, 600, 3), (746, 540, 3), (667, 380, 3), (810, 540, 3), (1050, 700, 3)]
        assert dp[0].shape == shapes[dp[1]]

def test_network_images():
    ds = NetworkImagesImple()
    ds.reset_state()

    assert ds.size() == 5
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels == [0, 1, 2, 3, 4]

def test_network_images_shuffle():
    ds = NetworkImagesImple(shuffle=True)
    ds.reset_state()

    assert ds.size() == 5
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels != [0, 1, 2, 3, 4]
    assert sorted(labels) == [0, 1, 2, 3, 4]

def test_network_images_partitioning_2():
    ds = NetworkImagesImple(shuffle=True).partitioning(2, 0)
    ds.reset_state()

    assert ds.size() == 3
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels != [0, 2, 4]
    assert sorted(labels) == [0, 2, 4]

    ds = NetworkImagesImple(shuffle=False).partitioning(2, 1)
    ds.reset_state()

    assert ds.size() == 2
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels == [1, 3]

def test_network_images_partitioning_3():
    ds = NetworkImagesImple().partitioning(3, 0)
    ds.reset_state()

    assert ds.size() == 2
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels == [0, 3]

    ds = NetworkImagesImple().partitioning(3, 1)
    ds.reset_state()

    assert ds.size() == 2
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels == [1, 4]

    ds = NetworkImagesImple().partitioning(3, 2)
    ds.reset_state()

    assert ds.size() == 1
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels == [2]

def test_network_images_partitioning_wroing():
    with pytest.raises(ValueError) as excinfo:
        ds = NetworkImagesImple().partitioning(2, 2)
    assert 'partition_index(=2) is must be a smaller than num_partition(=2)' == str(excinfo.value)

def test_network_images_parallel():
    ds = NetworkImagesImple().parallel(num_threads=3, buffer_size=3)
    ds.reset_state()

    assert ds.size() == 5
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels != [0, 1, 2, 3, 4]
    assert sorted(labels) == [0, 1, 2, 3, 4]

def test_network_images_partitioning_parallel():
    ds = NetworkImagesImple().partitioning(2, 0).parallel(num_threads=3, buffer_size=3)

    ds.reset_state()
    assert ds.size() == 3
    labels = []
    for dp in ds.get_data():
        NetworkImagesImple.test(dp)
        labels.append( dp[1] )
    assert labels != [0, 2, 4]
    assert sorted(labels) == [0, 2, 4]

