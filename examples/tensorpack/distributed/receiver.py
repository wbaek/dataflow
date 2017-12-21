import tensorpack.dataflow as df

if __name__ == '__main__':
    ds = df.RemoteDataZMQ('tcp://0.0.0.0:2222')
    df.TestDataSpeed(ds, size=50000).start()
