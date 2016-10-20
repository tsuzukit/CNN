import tensorflow as tf
import numpy as np

class Model:

    IMAGE_SIZE = 28
    BATCH_SIZE = 128

    def __init__(self, learning_rage):
        self.learning_rate = learning_rage

    def fit(self, training_data, training_label):
        num_labels = training_label.shape[1]

        graph = tf.Graph()
        with graph.as_default():
            tf_train_data_set = tf.placeholder(tf.float32, shape=(Model.BATCH_SIZE, Model.IMAGE_SIZE * Model.IMAGE_SIZE))
            tf_train_labels = tf.placeholder(tf.float32, shape=(Model.BATCH_SIZE, num_labels))

            weights = tf.Variable(tf.truncated_normal([Model.IMAGE_SIZE * Model.IMAGE_SIZE, num_labels]))
            biases = tf.Variable(tf.zeros([num_labels]))

            logits = tf.matmul(tf_train_data_set, weights) + biases
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            train_prediction = tf.nn.softmax(logits)

        with tf.Session(graph=graph) as session:
            init = tf.initialize_all_variables()
            session.run(init)

            for step in range(3000):
                offset = (step * Model.BATCH_SIZE) % (training_label.shape[0] - Model.BATCH_SIZE)
                batch_data = training_data[offset:(offset + Model.BATCH_SIZE), :]
                batch_labels = training_label[offset:(offset + Model.BATCH_SIZE), :]
                feed_dict = {
                    tf_train_data_set: batch_data,
                    tf_train_labels: batch_labels
                }
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                print('Training accuracy: %.1f%%' % Model._get_accuracy(predictions, batch_labels))

    @staticmethod
    def _get_accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
