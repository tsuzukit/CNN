
Deep Learning
=============

Assignment 3
------------

Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.

The goal of this assignment is to explore regularization techniques.


```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
```

First reload the data we generated in _notmist.ipynb_.


```python
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 28, 28) (200000,)
    Validation set (10000, 28, 28) (10000,)
    Test set (10000, 28, 28) (10000,)


Reformat into a shape that's more adapted to the models we're going to train:
- data as a flat matrix,
- labels as float 1-hot encodings.


```python
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 784) (200000, 10)
    Validation set (10000, 784) (10000, 10)
    Test set (10000, 784) (10000, 10)



```python
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
```


```python

```

---
Problem 1
---------

Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.

---


```python
batch_size = 128
H = 1024 # No. of hidden units

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, H]))
  biases1 = tf.Variable(tf.zeros([H]))
  weights2 = tf.Variable(tf.truncated_normal([H,  num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))

  # Regularization constants
  C = 5e-4
  R = tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2)+tf.nn.l2_loss(biases1)+tf.nn.l2_loss(biases2)
    
  # Training computation.
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  logits = tf.matmul(hidden1, weights2) + biases2
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + C * R)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
    
  # prediction of validation data set  
  valid_hidden1 =  tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_logits = tf.matmul(valid_hidden1, weights2) + biases2
  valid_prediction = tf.nn.softmax(valid_logits)
    
  # prediction of validation data set  
  test_hidden1 =  tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_logits = tf.matmul(test_hidden1, weights2) + biases2
  test_prediction = tf.nn.softmax(test_logits)
```


```python
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 503.708527
    Minibatch accuracy: 11.7%
    Validation accuracy: 29.4%
    Minibatch loss at step 500: 96.452362
    Minibatch accuracy: 80.5%
    Validation accuracy: 79.1%
    Minibatch loss at step 1000: 76.348495
    Minibatch accuracy: 78.9%
    Validation accuracy: 81.0%
    Minibatch loss at step 1500: 63.576965
    Minibatch accuracy: 85.9%
    Validation accuracy: 79.6%
    Minibatch loss at step 2000: 51.228325
    Minibatch accuracy: 89.1%
    Validation accuracy: 81.9%
    Minibatch loss at step 2500: 44.830116
    Minibatch accuracy: 83.6%
    Validation accuracy: 82.8%
    Minibatch loss at step 3000: 37.539890
    Minibatch accuracy: 84.4%
    Validation accuracy: 83.1%
    Test accuracy: 90.1%


---
Problem 2
---------
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?

---


```python
batch_size = 15
H = 1024 # No. of hidden units

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, H]))
  biases1 = tf.Variable(tf.zeros([H]))
  weights2 = tf.Variable(tf.truncated_normal([H,  num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))

  # Regularization constants
  C = 5e-4
  R = tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2)+tf.nn.l2_loss(biases1)+tf.nn.l2_loss(biases2)
    
  # Training computation.
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  logits = tf.matmul(hidden1, weights2) + biases2
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + C * R)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
    
  # prediction of validation data set  
  valid_hidden1 =  tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_logits = tf.matmul(valid_hidden1, weights2) + biases2
  valid_prediction = tf.nn.softmax(valid_logits)
    
  # prediction of validation data set  
  test_hidden1 =  tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_logits = tf.matmul(test_hidden1, weights2) + biases2
  test_prediction = tf.nn.softmax(test_logits)
```


```python
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 441.708557
    Minibatch accuracy: 13.3%
    Validation accuracy: 20.9%
    Minibatch loss at step 500: 260.450623
    Minibatch accuracy: 53.3%
    Validation accuracy: 59.7%
    Minibatch loss at step 1000: 762.149841
    Minibatch accuracy: 53.3%
    Validation accuracy: 52.2%
    Minibatch loss at step 1500: 143.843567
    Minibatch accuracy: 53.3%
    Validation accuracy: 54.4%
    Minibatch loss at step 2000: 145.771942
    Minibatch accuracy: 26.7%
    Validation accuracy: 55.0%
    Minibatch loss at step 2500: 84.397232
    Minibatch accuracy: 80.0%
    Validation accuracy: 66.4%
    Minibatch loss at step 3000: 67.593323
    Minibatch accuracy: 73.3%
    Validation accuracy: 70.8%
    Test accuracy: 78.8%


---
Problem 3
---------
Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.

What happens to our extreme overfitting case?

---


```python
batch_size = 128
H = 1024 # No. of hidden units
dropout = 0.5

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, H]))
  biases1 = tf.Variable(tf.zeros([H]))
  weights2 = tf.Variable(tf.truncated_normal([H,  num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
    
  # Training computation.
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  hidden1_dropped_out = tf.nn.dropout(hidden1, dropout)  
  logits = tf.matmul(hidden1_dropped_out, weights2) + biases2
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
    
  # prediction of validation data set  
  valid_hidden1 =  tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_logits = tf.matmul(valid_hidden1, weights2) + biases2
  valid_prediction = tf.nn.softmax(valid_logits)
    
  # prediction of validation data set  
  test_hidden1 =  tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_logits = tf.matmul(test_hidden1, weights2) + biases2
  test_prediction = tf.nn.softmax(test_logits)
```


```python
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 464.245667
    Minibatch accuracy: 8.6%
    Validation accuracy: 28.3%
    Minibatch loss at step 500: 37.572037
    Minibatch accuracy: 78.1%
    Validation accuracy: 79.3%
    Minibatch loss at step 1000: 18.565262
    Minibatch accuracy: 73.4%
    Validation accuracy: 80.0%
    Minibatch loss at step 1500: 14.029476
    Minibatch accuracy: 72.7%
    Validation accuracy: 77.3%
    Minibatch loss at step 2000: 11.072136
    Minibatch accuracy: 74.2%
    Validation accuracy: 79.2%
    Minibatch loss at step 2500: 6.818791
    Minibatch accuracy: 74.2%
    Validation accuracy: 80.0%
    Minibatch loss at step 3000: 3.357338
    Minibatch accuracy: 75.0%
    Validation accuracy: 80.0%
    Test accuracy: 87.3%


---
Problem 4
---------

Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).

One avenue you can explore is to add multiple layers.

Another one is to use learning rate decay:

    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
 
 ---



```python
batch_size = 128
H1 = 1024 # No. of hidden units
H2 = 512
dropout = 0.5

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, H1]))
  biases1 = tf.Variable(tf.zeros([H1]))
  weights2 = tf.Variable(tf.truncated_normal([H1, H2]))
  biases2 = tf.Variable(tf.zeros([H2]))
  weights3 = tf.Variable(tf.truncated_normal([H2,  num_labels]))
  biases3 = tf.Variable(tf.zeros([num_labels]))

  # Regularization constants
  C = 5e-4
  R = tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2)+tf.nn.l2_loss(biases1)+tf.nn.l2_loss(biases2)

  # Training computation.
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)
  logits = tf.matmul(hidden2, weights3) + biases3
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + C * R)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
    
  # prediction of validation data set  
  valid_hidden1 =  tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_hidden2 =  tf.nn.relu(tf.matmul(valid_hidden1, weights2) + biases2)
  valid_logits = tf.matmul(valid_hidden2, weights3) + biases3
  valid_prediction = tf.nn.softmax(valid_logits)
    
  # prediction of validation data set  
  test_hidden1 =  tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_hidden2 =  tf.nn.relu(tf.matmul(test_hidden1, weights2) + biases2)
  test_logits = tf.matmul(test_hidden2, weights3) + biases3
  test_prediction = tf.nn.softmax(test_logits)
```


```python
num_steps = 6001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 4162.037109
    Minibatch accuracy: 8.6%
    Validation accuracy: 14.3%
    Minibatch loss at step 500: 580.936768
    Minibatch accuracy: 78.9%
    Validation accuracy: 76.2%
    Minibatch loss at step 1000: 517.128174
    Minibatch accuracy: 78.9%
    Validation accuracy: 78.0%
    Minibatch loss at step 1500: 415.275391
    Minibatch accuracy: 83.6%
    Validation accuracy: 78.5%
    Minibatch loss at step 2000: 404.153992
    Minibatch accuracy: 84.4%
    Validation accuracy: 79.6%
    Minibatch loss at step 2500: 417.739563
    Minibatch accuracy: 80.5%
    Validation accuracy: 79.9%
    Minibatch loss at step 3000: 370.584869
    Minibatch accuracy: 82.8%
    Validation accuracy: 80.0%
    Minibatch loss at step 3500: 361.133057
    Minibatch accuracy: 83.6%
    Validation accuracy: 80.3%
    Minibatch loss at step 4000: 360.912720
    Minibatch accuracy: 81.2%
    Validation accuracy: 80.3%
    Minibatch loss at step 4500: 344.074219
    Minibatch accuracy: 84.4%
    Validation accuracy: 80.8%
    Minibatch loss at step 5000: 349.134979
    Minibatch accuracy: 83.6%
    Validation accuracy: 80.7%
    Minibatch loss at step 5500: 345.024597
    Minibatch accuracy: 78.9%
    Validation accuracy: 80.8%
    Minibatch loss at step 6000: 322.168488
    Minibatch accuracy: 80.5%
    Validation accuracy: 81.1%
    Test accuracy: 87.9%



```python

```
