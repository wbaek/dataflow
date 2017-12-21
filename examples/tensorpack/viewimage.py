import numpy as np
import tensorpack.dataflow as df
import cv2

if __name__ == '__main__':
    ds = df.dataset.Mnist('train')
    augmentors_variation = [
        df.imgaug.RandomApplyAug(df.imgaug.RandomResize((0.8, 1.2), (0.8, 1.2)), 0.5),
        df.imgaug.RandomApplyAug(df.imgaug.RotationAndCropValid(30), 0.5),
        df.imgaug.RandomApplyAug(df.imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01), 0.25),

        df.imgaug.Resize((32, 32)),
        df.imgaug.CenterPaste((36, 36)),
        df.imgaug.RandomCrop((32, 32)),

        df.imgaug.RandomOrderAug([
            df.imgaug.RandomApplyAug(df.imgaug.BrightnessScale((0.8, 1.2), clip=False), 0.5),
            df.imgaug.RandomApplyAug(df.imgaug.Contrast((0.8, 1.2), clip=False), 0.5),
            # df.imgaug.RandomApplyAug(df.imgaug.Saturation(0.4, rgb=False), 0.5),
        ]),
    ]
    augmentors_default = [
        df.imgaug.Resize((32, 32)),
        df.imgaug.MapImage(lambda x: x.reshape(32, 32, 1))
    ]
    # keep original image at index 1
    ds = df.MapData(ds, lambda datapoint: [datapoint[0], datapoint[0]] + datapoint[1:])
    ds = df.AugmentImageComponent(ds, augmentors_variation+augmentors_default)
    ds = df.AugmentImageComponent(ds, augmentors_default, index=1)
    ds = df.PrefetchData(ds, nr_prefetch=12, nr_proc=4)
    ds = df.PrintData(ds)
    ds = df.BatchData(ds, batch_size=32, remainder=False, use_list=True)
    ds = df.PrintData(ds)

    for minibatch in ds.get_data():
        images, originals, labels = minibatch
        image, original, label = images[0], originals[0], labels[0]
        name = '{:02d}'.format(label)

        cv2.namedWindow(name)
        cv2.moveWindow(name, 0, 25+128*label)

        display_image = np.concatenate((image, original), axis=1)
        cv2.imshow(name, display_image)
        cv2.waitKey(1)
