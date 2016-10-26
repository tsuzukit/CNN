import os
import sys
import tarfile
import struct
import numpy as np
import h5py
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from scipy import ndimage
from scipy.misc import imresize
import matplotlib.pyplot as plt


class SVHN:

    DEPTH = 255.0
    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32

    def __init__(self):
        self.last_download_percent = 0
        self._download_and_extract()
        SVHN._pickle_raw_data()
        SVHN._pickle_all_data()

        with open('resource/SVHN/all.pickle', 'rb') as f:
            dataset = pickle.load(f)
            self.training_dataset = dataset['train']['data']
            self.training_labels = dataset['train']['label']
            self.test_dataset = dataset['test']['data']
            self.test_labels = dataset['test']['label']

    def show_as_image(self, num=10):
        images = self.training_dataset
        labels = self.training_labels

        label_numbers = SVHN._get_label_numbers(labels)
        for index in range(num):
            print label_numbers[index]
            img = images[index, :, :]
            plt.figure()
            plt.imshow(img)  # display it
            plt.show()

    def get_histogram_data(self):
        labels = self.training_labels

        result = []
        label_numbers = SVHN._get_label_numbers(labels)
        for i in range(len(label_numbers)):
            order = len(label_numbers[i])
            result.append(order)

        return result

    def get_test_histogram_data(self):
        labels = self.test_labels

        result = []
        label_numbers = SVHN._get_label_numbers(labels)
        for i in range(len(label_numbers)):
            order = len(label_numbers[i])
            result.append(order)

        return result

    def get_reformatted_dataset(self):
        labels = self.training_labels
        images = self.training_dataset

        images = images.reshape((-1, images.shape[1], images.shape[2], 1))
        return images, labels

    def get_reformatted_test_dataset(self):
        labels = self.test_labels
        images = self.test_dataset

        images = images.reshape((-1, images.shape[1], images.shape[2], 1))
        return images, labels

    def _download_and_extract(self):
        self._maybe_download('train.tar.gz', 404141560)
        self._maybe_download('test.tar.gz', 276555967)
        SVHN._maybe_extract("train.tar.gz")
        SVHN._maybe_extract("test.tar.gz")

    @staticmethod
    def _pickle_raw_data():
        if not SVHN._has_pickle('train'):
            print ("start pickling training data")
            data = SVHN._load_data('resource/SVHN/train/digitStruct.mat')
            SVHN._maybe_pickle(data, "train")
        else:
            print ("training data is already pickled")

        if not SVHN._has_pickle('test'):
            print ("start pickling test data")
            data = SVHN._load_data('resource/SVHN/test/digitStruct.mat')
            SVHN._maybe_pickle(data, "test")
        else:
            print ("test data is already pickled")

    @staticmethod
    def _pickle_all_data():
        if SVHN._has_pickle('all'):
            print ("all data is already pickled")
            return

        with open('resource/SVHN/train.pickle', 'rb') as f:
            print ("loading training data from pickle")
            train_dataset = pickle.load(f)
            training_data = SVHN._load_images(train_dataset, "train")
            training_label = train_dataset['labels']

        with open('resource/SVHN/test.pickle', 'rb') as f:
            print ("loading test data from pickle")
            test_dataset = pickle.load(f)
            testing_data = SVHN._load_images(test_dataset, "test")
            testing_label = test_dataset['labels']

        print ("creating all pickle")
        with open('resource/SVHN/all.pickle', 'wb') as f:
            pickle.dump({
                'train': {'data': training_data, 'label': training_label},
                'test': {'data': testing_data, 'label': testing_label},
            }, f, pickle.HIGHEST_PROTOCOL)
            print ("all pickle is created")

    def _on_download_progress(self, count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent > self.last_download_percent:
            print('Download complete for %s' % percent + '%')
        self.last_download_percent = percent

    def _maybe_download(self, file_name, expected_bytes, force=False):
        self.last_download_percent = 0
        url = 'http://ufldl.stanford.edu/housenumbers/'
        file_path = "resource/SVHN/" + file_name
        if force or not os.path.exists(file_path):
            print('Attempting to download:', file_name)
            downloaded_file, _ = urlretrieve(url + file_name, file_path, reporthook=self._on_download_progress)
            print('\nDownload Complete')

        statinfo = os.stat(file_path)
        if statinfo.st_size == expected_bytes:
            print('Found and verified')
        else:
            print('Failed to verify. Please delete and try again')

    @staticmethod
    def _maybe_extract(file_name, force=False):
        folder_path = "resource/SVHN/"
        file_path = folder_path + file_name
        root = os.path.splitext(os.path.splitext(file_path)[0])[0]  # remove .tar.gz
        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, file_path))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(file_path)
            sys.stdout.flush()
            tar.extractall(folder_path)
            tar.close()
        if not os.path.exists(root+'/digitStruct.mat'):
            print("digitStruct.mat is missing")
        return root+'/digitStruct.mat'

    @staticmethod
    def _load_data(path):
        meta = h5py.File(path)

        digit_struct = meta['digitStruct']
        data_length = digit_struct['name'].shape[0]

        images = np.ndarray(shape=(data_length, ), dtype='|S18')
        labels = np.zeros((data_length, 6), dtype=float)
        labels.fill(0)
        tops = np.zeros((data_length, 6), dtype=float)
        heights = np.zeros((data_length, 6), dtype=float)
        widths = np.zeros((data_length, 6), dtype=float)
        lefts = np.zeros((data_length, 6), dtype=float)
        for i in xrange(data_length):
            images[i] = SVHN._get_label(meta, i)
            label = SVHN._get_attr(meta, i, 'label')
            top = SVHN._get_attr(meta, i, 'top')
            height = SVHN._get_attr(meta, i, 'height')
            width = SVHN._get_attr(meta, i, 'width')
            left = SVHN._get_attr(meta, i, 'left')

            labels[i, :label.shape[0]] = label
            tops[i, :top.shape[0]] = top
            heights[i, :height.shape[0]] = height
            widths[i, :width.shape[0]] = width
            lefts[i, :left.shape[0]] = left

        return labels, images, tops, heights, widths, lefts

    @staticmethod
    def _get_attr(meta, i, attr):
        ref = meta['digitStruct']['bbox'][i][0]
        d = meta[ref][attr].value.squeeze()
        if d.dtype == 'float64':
            return d.reshape(-1)
        return np.array([meta[x].value for x in d]).squeeze()

    @staticmethod
    def _get_label(meta, i):
        ref = meta['digitStruct']['name'][i][0]
        d = meta[ref].value.tostring()
        return d.replace('\x00', '')

    @staticmethod
    def _has_pickle(file_name):
        file_path = "resource/SVHN/" + file_name
        return os.path.exists(file_path + '.pickle')

    @staticmethod
    def _maybe_pickle(data_tuple, file_name, force=False):
        file_path = "resource/SVHN/" + file_name
        if os.path.exists(file_path + '.pickle') and not force:
            print('%s already present - Skipping pickling.' % struct)
        else:
            print('Pickling %s' % file_path + '.pickle')
            dataset = {
                'labels': data_tuple[0],
                'images': data_tuple[1],
                'tops': data_tuple[2],
                'heights': data_tuple[3],
                'widths': data_tuple[4],
                'lefts': data_tuple[5],
            }
        try:
            with open(file_path + '.pickle', 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', file_name + '.pickle', ':', e)

        return file_path + '.pickle'

    @staticmethod
    def _load_image(image_file, path, box):
        image_data = np.average(ndimage.imread(path + image_file), axis=2)
        image_data = image_data[box['y']:box['height'], box['x']:box['width']]
        image_data = imresize(image_data, (SVHN.IMAGE_WIDTH, SVHN.IMAGE_HEIGHT))
        image_data = (image_data.astype(float) - SVHN.DEPTH / 2) / SVHN.DEPTH
        return image_data

    @staticmethod
    def _load_images(dataset, target_name):
        images = dataset['images']
        tops = dataset['tops']
        widths = dataset['widths']
        heights = dataset['heights']
        lefts = dataset['lefts']

        path = 'resource/SVHN/' + target_name + "/"
        data = np.ndarray(shape=(images.shape[0], SVHN.IMAGE_WIDTH, SVHN.IMAGE_HEIGHT), dtype=np.float32)
        for i in range(data.shape[0]):
            count = dataset['labels'][i][dataset['labels'][i] > -1].shape[0]
            box = SVHN._get_box_coordinates(count, tops, heights, lefts, widths, i)
            if box is None:
                continue
            image = SVHN._load_image(images[i], path, box)
            data[i, :, :] = image

        return data

    @staticmethod
    def _get_box_coordinates(count, tops, heights, lefts, widths, index):
        top_heights = np.array([tops[index][:count], heights[index][:count]])
        left_widths = np.array([lefts[index][:count], widths[index][:count]])

        y = SVHN._get_top(top_heights)
        x = SVHN._get_left(left_widths)
        height = top_heights.sum(axis=0).max()
        width = left_widths.sum(axis=0).max()

        if y == float("inf") or x == float("inf"):
            return None

        return {
            "y": int(y),
            "x": int(x),
            "height": int(height),
            "width": int(width)
        }

    @staticmethod
    def _get_top(top_heights):
        result = float("inf")
        top_heights = top_heights[0]
        for top_height in top_heights:
            if top_height == 0.0:
                continue
            if top_height < result:
                result = top_height
        if result <= 0:
            result = 0
        return result

    @staticmethod
    def _get_left(left_width):
        result = float("inf")
        left_heights = left_width[0]
        for left_height in left_heights:
            if left_height == 0.0:
                continue
            if left_height < result:
                result = left_height
        if result <= 0:
            result = 0
        return result

    @staticmethod
    def _get_label_numbers(labels):
        result = []
        for label in labels:
            number_string = ""
            for digit in label:
                digit = str(int(digit))
                if digit == "10":
                    number_string += "0"
                elif digit != "0":
                    number_string += digit
            result.append(number_string)
        return result

