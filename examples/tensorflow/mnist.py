import tensorflow as tf
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
    ds = df.BatchData(ds, batch_size=128, remainder=False, use_list=False)

    # create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.ones([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(
        cross_entropy, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(3):
            for minibatch in ds.get_data():
                images, labels = minibatch
                _, step, l, a = sess.run([train_op, global_step, cross_entropy, accuracy],
                                         feed_dict={x: images, y_: labels})

                if step % 100 == 0:
                    print('epoch:{:02d}, step:{:06d}, loss:{:.3f}, accuracy:{:.3f}'.format(epoch, step, l, a))
