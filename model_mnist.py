import tensorflow as tf
import numpy as np


class CNN:

    IMAGE_SIZE = 28
    TRAINING_VALIDATION_RATIO = 0.95

    def __init__(self, learning_rage=0.05, batch_size=16, patch_size=5, depth=16, num_hidden=64, dropout=0.5, num_channels=1):
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
        num_labels = training_label.shape[1]

        graph = tf.Graph()
        with graph.as_default():
            # data
            tf_train_data_set = tf.placeholder(tf.float32, shape=(self.batch_size, CNN.IMAGE_SIZE, CNN.IMAGE_SIZE, self.num_channels))
            tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))
            tf_valid_data_set = tf.constant(validation_data)

            # variables
            layer1_weights = tf.Variable(tf.truncated_normal([self.patch_size, self.patch_size, self.num_channels, self.depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([self.depth]))
            layer2_weights = tf.Variable(tf.truncated_normal([self.patch_size, self.patch_size, self.depth, self.depth * 2], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth * 2]))
            layer3_weights = tf.Variable(tf.truncated_normal([CNN.IMAGE_SIZE // 4 * CNN.IMAGE_SIZE // 4 * self.depth * 2, self.num_hidden], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
            layer4_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

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
                hidden = tf.nn.dropout(hidden, self.dropout)
                return tf.matmul(hidden, layer4_weights) + layer4_biases

            # Training computation
            logits = model(tf_train_data_set)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Optimizer
            optimizer = tf.train.AdagradOptimizer(0.05).minimize(loss)

            # result
            training_prediction = tf.nn.softmax(logits)
            validation_prediction = tf.nn.softmax(model(tf_valid_data_set))

        with tf.Session(graph=graph) as session:
            # initialize
            init = tf.initialize_all_variables()
            session.run(init)

            # training
            for step in range(1000):
                batch_data, batch_labels = self._get_batch(step, training_data, training_label)
                feed_dict = {
                    tf_train_data_set: batch_data,
                    tf_train_labels: batch_labels
                }
                _, l, lg, predictions = session.run([optimizer, loss, logits, training_prediction], feed_dict=feed_dict)
                print("Step: %d" % step)
                print logits
                print('Training accuracy: %.1f%%' % CNN._get_accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % CNN._get_accuracy(validation_prediction.eval(), validation_label))

    @staticmethod
    def _get_accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    @staticmethod
    def _split_data(data, ratio):
        split = int(data.shape[0] * ratio)
        return np.vsplit(data, [split])

    def _get_batch(self, step, training_data, training_label):
        offset = (step * self.batch_size) % (training_label.shape[0] - self.batch_size)
        batch_data = training_data[offset:(offset + self.batch_size), :]
        batch_labels = training_label[offset:(offset + self.batch_size), :]
        return batch_data, batch_labels
