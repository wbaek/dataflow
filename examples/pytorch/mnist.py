import numpy as np
import torch
from torch.autograd import Variable
import tensorpack.dataflow as df


if __name__ == '__main__':
    # prepare dataset
    ds = df.dataset.Mnist('train')
    augmentors_variation = [
        df.imgaug.Resize((28, 28)),
        df.imgaug.CenterPaste((32, 32)),
        df.imgaug.RandomCrop((28, 28)),

        df.imgaug.MapImage(lambda v: v.reshape(784))
    ]
    ds = df.AugmentImageComponent(ds, augmentors_variation)
    ds = df.PrefetchData(ds, nr_prefetch=12, nr_proc=4)
    ds = df.PrintData(ds)
    ds = df.BatchData(ds, batch_size=128, remainder=False, use_list=False)
    ds = df.PrintData(ds)

    # create the model
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = torch.nn.Linear(784, 10)

        def forward(self, x):
            x = self.fc(x)
            return torch.nn.functional.log_softmax(x)
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # train
    step = 0
    for epoch in range(3):
        for minibatch in ds.get_data():
            images = Variable(torch.from_numpy(minibatch[0]))
            labels = Variable(torch.from_numpy(minibatch[1].astype(np.int64)))

            optimizer.zero_grad()
            output = model(images)
            loss = torch.nn.functional.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

            l = loss.data[0]

            pred = output.data.max(1, keepdim=True)[1]
            a = pred.eq(labels.data.view_as(pred)).cpu().sum() / 128.0
            step += 1

        print('epoch:{:02d}, step:{:06d}, loss:{:.3f}, accuracy:{:.3f}'.format(epoch, step, l, a))
