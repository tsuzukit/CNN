import tensorflow as tf
import numpy as np
import csv
import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


class CNN:

    # 0 to 9 plus undefined
    NUM_LABELS = 11
    NUM_DIGITS = 6
    IMAGE_SIZE = 32
    NUM_CHANNELS = 1
    TRAINING_VALIDATION_RATIO = 0.99
    TRAINING_MODE = 0
    TRAINING_VALIDATION_MODE = 1
    TEST_MODE = 2
    PREDICTION_MODE = 3
    HIDDEN_LAYER1_NUM = 128
    HIDDEN_LAYER2_NUM = 64

    def __init__(self,
                 learning_rage=0.1,
                 batch_size=32,
                 patch_size=8,
                 dropout=0.8,
                 training_num=60000):
        """
        :param learning_rage: learning rate
        :param batch_size: number of data used in training at each iteration
        :param patch_size: patch size of convolution layer
        :param dropout: ratio of keeping data at dropout
        :param training_num: number of iteration during training
        """

        self.learning_rate = learning_rage
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.dropout = dropout
        self.training_num = training_num

    def fit(self, training_data, training_label):
        """
        :param training_data: training data
        :param training_label: training label
        Train the model
        """

        training_data, validation_data = CNN._split_data(training_data, CNN.TRAINING_VALIDATION_RATIO)
        training_label, validation_label = CNN._split_data(training_label, CNN.TRAINING_VALIDATION_RATIO)
        self._setup_and_run_graph(training_data=training_data,
                                  training_label=training_label,
                                  validation_data=validation_data,
                                  validation_label=validation_label,
                                  mode=CNN.TRAINING_VALIDATION_MODE)

    def test(self, test_data, test_label):
        """
        :param test_data: test data
        :param test_label: test label
        Test the model
        """

        self._setup_and_run_graph(training_data=None,
                                  training_label=None,
                                  validation_data=test_data,
                                  validation_label=test_label,
                                  mode=CNN.TEST_MODE)

    def predict(self, test_data):
        """
        :param test_data: test data
        :return numbers predicted
        Predict numbers in the picture
        """

        return self._setup_and_run_graph(training_data=None,
                                         training_label=None,
                                         validation_data=test_data,
                                         validation_label=None,
                                         mode=CNN.PREDICTION_MODE)

    def _setup_and_run_graph(self, training_data, training_label, validation_data, validation_label, mode):
        """
        :param training_data: training data
        :param training_label: training label
        :param validation_data: validation or test data
        :param validation_label: validation or test label
        :param mode: when in test and prediction mode, the model uses parameters which are saved after training
        """

        num_labels = CNN.NUM_LABELS
        num_digits = CNN.NUM_DIGITS

        graph = tf.Graph()
        with graph.as_default():
            # data
            tf_train_data_set = tf.placeholder(tf.float32, shape=(self.batch_size, CNN.IMAGE_SIZE, CNN.IMAGE_SIZE, CNN.NUM_CHANNELS))
            tf_train_labels = tf.placeholder(tf.int32, shape=(self.batch_size, num_digits))
            tf_valid_data_set = tf.constant(validation_data)

            # variables
            layer1_weights = tf.Variable(tf.truncated_normal([self.patch_size, self.patch_size, CNN.NUM_CHANNELS, 5], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([5]))
            layer2_weights = tf.Variable(tf.truncated_normal([self.patch_size / 2, self.patch_size / 2, 5, 55], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[55]))
            layer3_weights = tf.Variable(tf.truncated_normal([self.patch_size / 4, self.patch_size / 4, 55, 110], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[110]))
            layer4_weights = tf.Variable(tf.truncated_normal([self.patch_size // 8, self.patch_size // 8, 110, 220], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[220]))

            full_layer1_weights = tf.Variable(tf.truncated_normal([CNN.IMAGE_SIZE // 16 * CNN.IMAGE_SIZE // 16 * 220, CNN.HIDDEN_LAYER1_NUM], stddev=0.1))
            full_layer1_biases = tf.Variable(tf.constant(1.0, shape=[CNN.HIDDEN_LAYER1_NUM]))
            full_layer2_weights = tf.Variable(tf.truncated_normal([CNN.HIDDEN_LAYER1_NUM, CNN.HIDDEN_LAYER2_NUM], stddev=0.1))
            full_layer2_biases = tf.Variable(tf.constant(1.0, shape=[CNN.HIDDEN_LAYER2_NUM]))

            s1_weights = tf.Variable(tf.truncated_normal([CNN.HIDDEN_LAYER2_NUM, num_labels], stddev=0.1))
            s1_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s2_weights = tf.Variable(tf.truncated_normal([CNN.HIDDEN_LAYER2_NUM, num_labels], stddev=0.1))
            s2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s3_weights = tf.Variable(tf.truncated_normal([CNN.HIDDEN_LAYER2_NUM, num_labels], stddev=0.1))
            s3_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s4_weights = tf.Variable(tf.truncated_normal([CNN.HIDDEN_LAYER2_NUM, num_labels], stddev=0.1))
            s4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s5_weights = tf.Variable(tf.truncated_normal([CNN.HIDDEN_LAYER2_NUM, num_labels], stddev=0.1))
            s5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

            # Model
            def model(data):
                conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer1_biases)
                pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                dropout = tf.nn.dropout(pool, self.dropout)
                norm = tf.nn.lrn(dropout, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75)

                conv = tf.nn.conv2d(norm, layer2_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer2_biases)
                pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                dropout = tf.nn.dropout(pool, self.dropout)
                norm = tf.nn.lrn(dropout, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75)

                conv = tf.nn.conv2d(norm, layer3_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer3_biases)
                pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                dropout = tf.nn.dropout(pool, self.dropout)
                norm = tf.nn.lrn(dropout, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75)

                conv = tf.nn.conv2d(norm, layer4_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer4_biases)
                pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                dropout = tf.nn.dropout(pool, self.dropout)
                norm = tf.nn.lrn(dropout, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75)

                shape = norm.get_shape().as_list()
                reshape = tf.reshape(norm, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, full_layer1_weights) + full_layer1_biases)
                hidden = tf.nn.relu(tf.matmul(hidden, full_layer2_weights) + full_layer2_biases)

                dropout = tf.nn.dropout(hidden, self.dropout)

                logits1 = tf.matmul(dropout, s1_weights) + s1_biases
                logits2 = tf.matmul(dropout, s2_weights) + s2_biases
                logits3 = tf.matmul(dropout, s3_weights) + s3_biases
                logits4 = tf.matmul(dropout, s4_weights) + s4_biases
                logits5 = tf.matmul(dropout, s5_weights) + s5_biases
                return [logits1, logits2, logits3, logits4, logits5]

            # Regularization constants
            C = 5e-4
            R = tf.nn.l2_loss(layer1_weights)+tf.nn.l2_loss(layer1_biases) + \
                tf.nn.l2_loss(layer2_weights)+tf.nn.l2_loss(layer2_biases) + \
                tf.nn.l2_loss(layer3_weights)+tf.nn.l2_loss(layer3_biases) + \
                tf.nn.l2_loss(layer4_weights)+tf.nn.l2_loss(layer4_biases) + \
                tf.nn.l2_loss(full_layer1_weights)+tf.nn.l2_loss(full_layer1_biases) + \
                tf.nn.l2_loss(full_layer2_weights)+tf.nn.l2_loss(full_layer2_biases) + \
                tf.nn.l2_loss(s1_weights)+tf.nn.l2_loss(s1_biases) + \
                tf.nn.l2_loss(s2_weights)+tf.nn.l2_loss(s2_biases) + \
                tf.nn.l2_loss(s3_weights)+tf.nn.l2_loss(s3_biases) + \
                tf.nn.l2_loss(s4_weights)+tf.nn.l2_loss(s4_biases) + \
                tf.nn.l2_loss(s5_weights)+tf.nn.l2_loss(s5_biases)

            # Training computation
            logits = model(tf_train_data_set)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], tf_train_labels[:, 0])) + \
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[1], tf_train_labels[:, 1])) + \
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[2], tf_train_labels[:, 2])) + \
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[3], tf_train_labels[:, 3])) + \
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[4], tf_train_labels[:, 4])) + \
                   C * R

            # Optimizer
            optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

            # result
            training_prediction = tf.pack([tf.nn.softmax(logits[0]),
                                           tf.nn.softmax(logits[1]),
                                           tf.nn.softmax(logits[2]),
                                           tf.nn.softmax(logits[3]),
                                           tf.nn.softmax(logits[4])])

            validation_prediction = tf.pack([tf.nn.softmax(model(tf_valid_data_set)[0]),
                                             tf.nn.softmax(model(tf_valid_data_set)[1]),
                                             tf.nn.softmax(model(tf_valid_data_set)[2]),
                                             tf.nn.softmax(model(tf_valid_data_set)[3]),
                                             tf.nn.softmax(model(tf_valid_data_set)[4])])

            # to save model trained
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as session:
            if mode == CNN.PREDICTION_MODE:
                saver.restore(session, "result/SVHN_MODEL.ckpt")
                predictions = session.run(validation_prediction)
                prediction_numbers = CNN._get_prediction_numbers(predictions)
                return prediction_numbers

            if mode == CNN.TEST_MODE:
                saver.restore(session, "result/SVHN_MODEL.ckpt")
                predictions = session.run(validation_prediction)
                accuracy = CNN._get_accuracy(predictions, validation_label[:, 0:5])
                precision = CNN._get_precision(predictions, validation_label[:, 0:5])
                recall = CNN._get_recall(predictions, validation_label[:, 0:5])
                print('Accuracy: %.1f%%' % accuracy)
                print('Precision: %.1f' % precision)
                print('Recall: %.1f' % recall)
                return

            # for training

            # initialize
            init = tf.initialize_all_variables()
            session.run(init)

            # for statis
            stats = []

            # training
            start = time.time()
            for step in range(self.training_num):

                batch_data, batch_labels = self._get_batch(step, training_data, training_label)
                feed_dict = {
                    tf_train_data_set: batch_data,
                    tf_train_labels: batch_labels
                }
                _, l, tf_l, lg, predictions = session.run([optimizer, loss, tf_train_labels, logits, training_prediction], feed_dict=feed_dict)

                if step % 5 == 0:
                    training_accuracy = CNN._get_accuracy(predictions, batch_labels[:, 0:5])
                    print("Step: %d" % step)
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Training accuracy: %.1f%%' % training_accuracy)

                    if mode == CNN.TRAINING_VALIDATION_MODE:
                        validation_predictions = validation_prediction.eval()
                        validation_accuracy = CNN._get_accuracy(validation_predictions, validation_label[:, 0:5])
                        print('Validation accuracy: %.1f%%' % validation_accuracy)
                    else:
                        validation_accuracy = 0

                    if step % 500 == 0:
                        # for stats
                        stats_data = {"step": step,
                                      "training_accuracy": training_accuracy,
                                      "validation_accuracy": validation_accuracy}
                        stats.append(stats_data)

            elapsed_time = time.time() - start
            print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

            # save stats
            keys = stats[0].keys()
            with open('result/stats.csv', 'wb') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(stats)

            # save model
            saver.save(session, "result/SVHN_MODEL.ckpt")

    @staticmethod
    def _get_accuracy(predictions, labels):
        """
        :param predictions: predicted labels
        :param labels: true labels
        :return: accuracy in percent
        """

        label_numbers = CNN._get_label_numbers(labels)
        prediction_numbers = CNN._get_prediction_numbers(predictions)
        correct = 0
        correct_1 = 0
        correct_2 = 0
        correct_3 = 0
        correct_4 = 0
        correct_5 = 0
        correct_6 = 0
        for i, label in enumerate(label_numbers):
            if label == prediction_numbers[i]:
                correct += 1
                if len(label) == 1:
                    correct_1 += 1
                if len(label) == 2:
                    correct_2 += 1
                if len(label) == 3:
                    correct_3 += 1
                if len(label) == 4:
                    correct_4 += 1
                if len(label) == 5:
                    correct_5 += 1
                if len(label) == 6:
                    correct_6 += 1

        print correct_1, correct_2, correct_3, correct_4, correct_5, correct_6

        return 100.0 * correct / len(label_numbers)

    @staticmethod
    def _get_precision(predictions, labels):
        """
        :param predictions: predicted labels
        :param labels: true labels
        :return: precision
        """

        label_numbers = CNN._get_label_numbers(labels)
        prediction_numbers = CNN._get_prediction_numbers(predictions)
        return precision_score(label_numbers, prediction_numbers, average='micro')

    @staticmethod
    def _get_recall(predictions, labels):
        """
        :param predictions: predicted labels
        :param labels: true labels
        :return: recall
        """

        label_numbers = CNN._get_label_numbers(labels)
        prediction_numbers = CNN._get_prediction_numbers(predictions)
        return recall_score(label_numbers, prediction_numbers, average='micro')

    @staticmethod
    def _get_confusion_matrix(predictions, labels):
        """
        :param predictions: predicted labels
        :param labels: true labels
        :return: confusion matrix
        """

        label_numbers = CNN._get_label_numbers(labels)
        prediction_numbers = CNN._get_prediction_numbers(predictions)
        unique = list(set(label_numbers))
        sorted_list = sorted(unique)
        return confusion_matrix(label_numbers, prediction_numbers, labels=sorted_list)

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

    @staticmethod
    def _get_prediction_numbers(predictions):
        labels = np.argmax(predictions, 2).T
        return CNN._get_label_numbers(labels)

    @staticmethod
    def _split_data(data, ratio):
        split = int(data.shape[0] * ratio)
        return np.vsplit(data, [split])

    def _get_batch(self, step, training_data, training_label):
        offset = (step * self.batch_size) % (training_label.shape[0] - self.batch_size)
        batch_data = training_data[offset:(offset + self.batch_size), :]
        batch_labels = training_label[offset:(offset + self.batch_size), :]
        return batch_data, batch_labels
