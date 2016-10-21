import struct
import numpy as np
import matplotlib.pyplot as plt

class MNIST:

    TYPE_TRAINING = 0
    TYPE_TEST = 1

    def __init__(self, data_type, num_channels=1):
        self.data_type = data_type
        self.num_channels = num_channels

        label_file_path, image_file_path = self._get_source_file_paths()
        self.labels = MNIST._read_labels(label_file_path)
        self.images = MNIST._read_images(image_file_path)

    def get_reformatted_dataset(self):
        labels = self.labels
        images = self.images

        images = images.reshape((-1, images.shape[1], images.shape[2], self.num_channels)).astype(np.float32)
        labels = (np.arange(10) == labels[:, None]).astype(np.float32)
        return images, labels

    @staticmethod
    def _read_labels(label_file_path):

        with open(label_file_path, 'r') as f:

            # header (two 4B integers, magic number(2049) & number of items)
            header = f.read(8)
            mn, num = struct.unpack('>2i', header)  # MSB first (bigendian)
            assert mn == 2049

            # labels (unsigned byte)
            label = np.array(struct.unpack('>%dB' % num, f.read()), dtype=int)

        return label

    @staticmethod
    def _read_images(image_file_path):

        with open(image_file_path, 'r') as f:

            # header (four 4B integers, magic number(2051), #images, #rows, and #cols
            header = f.read(16)
            mn, num, nrow, ncol = struct.unpack('>4i', header) # MSB first (bigendian)
            assert mn == 2051

            # pixels (unsigned byte)
            pixel = np.empty((num, nrow, ncol))
            npixel = nrow * ncol
            for i in range(num):
                buf = struct.unpack('>%dB' % npixel, f.read( npixel))
                pixel[i, :, :] = np.asarray(buf).reshape((nrow, ncol))

        return pixel

    def _get_source_file_names(self):
        if self.data_type == MNIST.TYPE_TRAINING:
            return "train-labels-idx1-ubyte", "train-images-idx3-ubyte"
        elif self.data_type == MNIST.TYPE_TEST:
            return "t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte"
        else:
            raise ValueError('Undefined data type')

    def _get_source_file_paths(self):
        label_file_name, image_file_name = self._get_source_file_names()
        return "resource/MNIST/" + label_file_name, "resource/MNIST/" + image_file_name

    def show_as_image(self, nx=10, ny=10, gap=4):
        images = self.images

        nrow, ncol = images.shape[1:]
        width = nx * (ncol + gap) + gap
        height = ny * (nrow + gap) + gap
        img = np.zeros((height, width), dtype=int) + 128

        for iy in range(ny):
            lty = iy * (nrow + gap) + gap
            for ix in range(nx):
                ltx = ix * (ncol + gap) + gap
                img[lty:lty+nrow, ltx:ltx+ncol] = images[iy*nx+ix]

        plt.figure()
        plt.imshow(img)  # display it
        plt.show()
