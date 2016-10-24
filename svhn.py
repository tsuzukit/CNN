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
        SVHN.download_and_extract()
        SVHN.pickle_raw_data()
        SVHN.pickle_all_data()

        with open('resource/SVHN/all.pickle', 'rb') as f:
            dataset = pickle.load(f)
            self.training_dataset = dataset['train']['data']
            self.training_labels = dataset['train']['label']
            self.test_dataset = dataset['test']['data']
            self.test_labels = dataset['test']['label']

    @staticmethod
    def download_and_extract():
        SVHN.maybe_download('train.tar.gz', 404141560)
        SVHN.maybe_download('test.tar.gz', 276555967)
        SVHN.maybe_extract("train.tar.gz")
        SVHN.maybe_extract("test.tar.gz")

    @staticmethod
    def pickle_raw_data():
        if not SVHN.has_pickle('train'):
            print ("train data is already pickled")
            train_tuple = SVHN.load_data('resource/SVHN/train/digitStruct.mat')
            SVHN.maybe_pickle(train_tuple, "train")

        if not SVHN.has_pickle('test'):
            print ("test data is already pickled")
            test_tuple = SVHN.load_data('resource/SVHN/test/digitStruct.mat')
            SVHN.maybe_pickle(test_tuple, "test")

    @staticmethod
    def pickle_all_data():
        if SVHN.has_pickle('all'):
            print ("all data is already pickled")
            return

        with open('resource/SVHN/train.pickle', 'rb') as f:
            print ("loading training data from pickle")
            train_dataset = pickle.load(f)
            training_data = SVHN.load_images(train_dataset, "train")
            training_label = train_dataset['labels']

        with open('resource/SVHN/test.pickle', 'rb') as f:
            print ("loading test data from pickle")
            test_dataset = pickle.load(f)
            testing_data = SVHN.load_images(test_dataset, "test")
            testing_label = test_dataset['labels']

        print ("creating all pickle")
        with open('resource/SVHN/all.pickle', 'wb') as f:
            pickle.dump({
                'train': {'data': training_data, 'label': training_label},
                'test': {'data': testing_data, 'label': testing_label},
            }, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def download_progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print percent

    @staticmethod
    def maybe_download(file_name, expected_bytes, force=False):
        url = 'http://ufldl.stanford.edu/housenumbers/'
        file_path = "resource/SVHN/" + file_name
        if force or not os.path.exists(file_path):
            print('Attempting to download:', file_name)
            downloaded_file, _ = urlretrieve(url + file_name, file_path, reporthook=SVHN.download_progress_hook)
            print('\nDownload Complete!')

        statinfo = os.stat(file_path)
        if statinfo.st_size == expected_bytes:
            print('Found and verified')
        else:
            print('Failed to verify. Please delete and try again')

    @staticmethod
    def maybe_extract(file_name, force=False):
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
    def load_data(path):
        c = h5py.File(path)

        images = a = np.ndarray(shape=(c['digitStruct']['name'].shape[0], ), dtype='|S15')
        labels = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
        labels.fill(10)
        tops = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
        heights = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
        widths = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
        lefts = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
        for i in xrange(c['digitStruct']['name'].shape[0]):
            images[i] = SVHN.get_label(c, i)
            l = SVHN.get_attr(c, i, 'label')
            t = SVHN.get_attr(c, i, 'top')
            h = SVHN.get_attr(c, i, 'height')
            w = SVHN.get_attr(c, i, 'width')
            le = SVHN.get_attr(c, i, 'left')

            labels[i, :l.shape[0]] = l
            tops[i, :t.shape[0]] = t
            heights[i, :h.shape[0]] = h
            widths[i, :w.shape[0]] = w
            lefts[i, :le.shape[0]] = le

            if (i % 5000 == 0):
                print(i, "elapsed")

        return labels, images, tops, heights, widths, lefts

    @staticmethod
    def get_attr(c, i, attr):
        d = c[c['digitStruct']['bbox'][i][0]][attr].value.squeeze()
        if d.dtype == 'float64':
            return d.reshape(-1)
        return np.array([c[x].value for x in d]).squeeze()

    @staticmethod
    def get_label(c, i):
        d = c[c['digitStruct']['name'][i][0]].value.tostring()
        return d.replace('\x00', '')

    @staticmethod
    def has_pickle(file_name):
        file_path = "resource/SVHN/" + file_name
        return os.path.exists(file_path + '.pickle')

    @staticmethod
    def maybe_pickle(data_tuple, file_name, force=False):
        file_path = "resource/SVHN/" + file_name
        if os.path.exists(file_path + '.pickle') and not force:
            print('%s already present - Skipping pickling.' % struct)
        else:
            print('Pickling %s.' % file_path + '.pickle')
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
    def load_image(image_file, path, box):
        image_data = np.average(ndimage.imread(path + image_file), axis=2)
        if box['minTop'] <= 0:
            box['minTop'] = 0
        if box['minLeft'] <= 0:
            box['minLeft'] = 0
        image_data = image_data[box['minTop']:box['maxTopHeight'], box['minLeft']:box['maxLeftWidth']]
        image_data = imresize(image_data, (SVHN.IMAGE_WIDTH, SVHN.IMAGE_HEIGHT))
        image_data = (image_data.astype(float) - SVHN.DEPTH / 2) / SVHN.DEPTH

        return image_data

    @staticmethod
    def load_images(dataset, target_name):
        images = dataset['images']
        tops = dataset['tops']
        widths = dataset['widths']
        heights = dataset['heights']
        lefts = dataset['lefts']

        data = np.ndarray(shape=(images.shape[0], SVHN.IMAGE_WIDTH, SVHN.IMAGE_HEIGHT), dtype=np.float32)
        for i in range(data.shape[0]):
            path = 'resource/SVHN/' + target_name + "/"
            chr_count = dataset['labels'][i][dataset['labels'][i] > -1].shape[0]
            top_heights = np.array([tops[i][:chr_count], heights[i][:chr_count]])
            left_widths = np.array([lefts[i][:chr_count], widths[i][:chr_count]])

            top = SVHN.get_top(top_heights)
            left = SVHN.get_left(left_widths)
            if top == float("inf") or left == float("inf"):
                continue

            image = SVHN.load_image(images[i], path, {
                    "minTop": int(top),
                    "minLeft": int(left),
                    "maxTopHeight": int(top_heights.sum(axis=0).max()),
                    "maxLeftWidth": int(left_widths.sum(axis=0).max())
            })
            data[i, :, :] = image

        return data

    @staticmethod
    def get_top(top_heights):
        result = float("inf")
        top_heights = top_heights[0]
        for top_height in top_heights:
            if top_height == 0.0:
                continue
            if top_height < result:
                result = top_height
        return result

    @staticmethod
    def get_left(left_width):
        result = float("inf")
        left_heights = left_width[0]
        for left_height in left_heights:
            if left_height == 0.0:
                continue
            if left_height < result:
                result = left_height
        return result

    def show_as_image(self, num=10):
        images = self.training_dataset
        labels = self.training_labels

        for index in range(num):
            print labels[index]
            img = images[index, :, :]
            plt.figure()
            plt.imshow(img)  # display it
            plt.show()

    def get_histogram_data(self):
        labels = self.training_labels

        result = {}
        label_numbers = SVHN._get_label_numbers(labels)
        for i in range(len(label_numbers)):
            order = str(len(label_numbers[i]))
            if order in result.keys():
                result[order] += 1
            else:
                result[order] = 1

        print result

    @staticmethod
    def _get_label_numbers(labels):
        result = []
        for label in labels:
            number_string = ""
            for digit in label:
                digit = str(int(digit))
                if digit != "10":
                    number_string += digit
            result.append(number_string)
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
