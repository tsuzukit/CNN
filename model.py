import tensorflow as tf
import numpy as np


class Model:

    IMAGE_SIZE = 28

    def __init__(self, learning_rage=0.05, batch_size=16, patch_size=5, depth=16, num_hidden=64, dropout=0.5):
        self.learning_rate = learning_rage
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_hidden = num_hidden
        self.dropout = dropout

    def fit(self, training_data, training_label):
        training_data, validation_data = Model._split_data(training_data, 0.95)
        training_label, validation_label = Model._split_data(training_label, 0.95)
        num_labels = training_label.shape[1]

        graph = tf.Graph()
        with graph.as_default():
            # data
            tf_train_data_set = tf.placeholder(tf.float32, shape=(self.batch_size, Model.IMAGE_SIZE * Model.IMAGE_SIZE))
            tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))
            tf_valid_data_set = tf.constant(validation_data)

            # variables
            weights = tf.Variable(tf.truncated_normal([Model.IMAGE_SIZE * Model.IMAGE_SIZE, num_labels]))
            biases = tf.Variable(tf.zeros([num_labels]))

            # calculation
            logits = tf.matmul(tf_train_data_set, weights) + biases
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # result
            training_prediction = tf.nn.softmax(logits)
            validation_prediction = tf.nn.softmax(tf.matmul(tf_valid_data_set, weights) + biases)

        with tf.Session(graph=graph) as session:
            # initialize
            init = tf.initialize_all_variables()
            session.run(init)

            # training
            for step in range(3000):
                batch_data, batch_labels = self._get_batch(step, training_data, training_label)
                feed_dict = {
                    tf_train_data_set: batch_data,
                    tf_train_labels: batch_labels
                }
                _, l, predictions = session.run([optimizer, loss, training_prediction], feed_dict=feed_dict)
                print('Training accuracy: %.1f%%' % Model._get_accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % Model._get_accuracy(validation_prediction.eval(), validation_label))

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
