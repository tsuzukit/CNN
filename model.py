import tensorflow as tf
import numpy as np
import csv


class CNN:

    IMAGE_SIZE = 32
    TRAINING_VALIDATION_RATIO = 0.99

    def __init__(self, learning_rage=0.05, batch_size=64, patch_size=5, depth=16, num_hidden=64, dropout=0.5, num_channels=1):
        self.learning_rate = learning_rage
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.num_channels = num_channels

    def fit(self, training_data, training_label):
        training_data, validation_data = CNN._split_data(training_data, CNN.TRAINING_VALIDATION_RATIO)
        training_label, validation_label = CNN._split_data(training_label, CNN.TRAINING_VALIDATION_RATIO)
        num_labels = 11
        num_digits = 6

        graph = tf.Graph()
        with graph.as_default():
            # data
            tf_train_data_set = tf.placeholder(tf.float32, shape=(self.batch_size, CNN.IMAGE_SIZE, CNN.IMAGE_SIZE, self.num_channels))
            tf_train_labels = tf.placeholder(tf.int32, shape=(self.batch_size, num_digits))
            tf_valid_data_set = tf.constant(validation_data)

            # variables
            layer1_weights = tf.Variable(tf.truncated_normal([self.patch_size, self.patch_size, self.num_channels, self.depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([self.depth]))
            layer2_weights = tf.Variable(tf.truncated_normal([self.patch_size, self.patch_size, self.depth, self.depth * 2], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth * 2]))
            layer3_weights = tf.Variable(tf.truncated_normal([CNN.IMAGE_SIZE // 4 * CNN.IMAGE_SIZE // 4 * self.depth * 2, self.num_hidden], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))

            s1_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1))
            s1_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s2_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1))
            s2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s3_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1))
            s3_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s4_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1))
            s4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
            s5_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1))
            s5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

            # Model
            def model(data):
                conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
                pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                hidden = tf.nn.relu(pool + layer1_biases)
                conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
                pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                hidden = tf.nn.relu(pool + layer2_biases)
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

                logits1 = tf.matmul(hidden, s1_weights) + s1_biases
                logits2 = tf.matmul(hidden, s2_weights) + s2_biases
                logits3 = tf.matmul(hidden, s3_weights) + s3_biases
                logits4 = tf.matmul(hidden, s4_weights) + s4_biases
                logits5 = tf.matmul(hidden, s5_weights) + s5_biases
                return [logits1, logits2, logits3, logits4, logits5]

            # Training computation
            logits = model(tf_train_data_set)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], tf_train_labels[:, 1])) +\
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[1], tf_train_labels[:, 2])) +\
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[2], tf_train_labels[:, 3])) +\
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[3], tf_train_labels[:, 4])) +\
                   tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[4], tf_train_labels[:, 5]))

            # Optimizer
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

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
            # initialize
            init = tf.initialize_all_variables()
            session.run(init)

            # for statis
            stats = []

            # training
            for step in range(10):

                batch_data, batch_labels = self._get_batch(step, training_data, training_label)
                feed_dict = {
                    tf_train_data_set: batch_data,
                    tf_train_labels: batch_labels
                }
                _, l, lg, predictions = session.run([optimizer, loss, logits, training_prediction], feed_dict=feed_dict)
                training_accuracy = CNN._get_accuracy(predictions, batch_labels[:, 1:6])
                validation_accuracy = CNN._get_accuracy(validation_prediction.eval(), validation_label[:, 1:6])
                print("Step: %d" % step)
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % training_accuracy)
                print('Validation accuracy: %.1f%%' % validation_accuracy)

                # for stats
                stats_data = {"step": step,
                              "training_accuracy": training_accuracy,
                              "validation_accuracy": validation_accuracy}
                stats.append(stats_data)

            # save stats
            keys = stats[0].keys()
            with open('result.csv', 'wb') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(stats)

            # save model
            saver.save(session, "SVHN_MODEL.ckpt")

    @staticmethod
    def _get_accuracy(predictions, labels):
        label_numbers = CNN._get_label_numbers(labels)
        prediction_numbers = CNN._get_prediction_numbers(predictions)
        correct = 0
        for i, label in enumerate(label_numbers):
            if label == prediction_numbers[i]:
                # for debug
                print label, prediction_numbers[i]
                correct += 1
        return 100.0 * correct / len(label_numbers)

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

    @staticmethod
    def _get_prediction_numbers(predictions):
        labels = np.argmax(predictions, 2).T
        result = []
        for label in labels:
            number_string = ""
            for digit in label:
                digit = str(int(digit))
                if digit != "10":
                    number_string += digit
            result.append(number_string)
        return result

    @staticmethod
    def _split_data(data, ratio):
        split = int(data.shape[0] * ratio)
        return np.vsplit(data, [split])

    def _get_batch(self, step, training_data, training_label):
        offset = (step * self.batch_size) % (training_label.shape[0] - self.batch_size)
        batch_data = training_data[offset:(offset + self.batch_size), :]
        batch_labels = training_label[offset:(offset + self.batch_size), :]
        return batch_data, batch_labels
