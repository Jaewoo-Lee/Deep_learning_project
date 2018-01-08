import tensorflow as tf
import os

# same params and variables initialization as log reg.
X = tf.placeholder("float", [None, 784], name="input")
W1 = tf.Variable(tf.zeros([784, 100]), name="weights1")
b1 = tf.Variable(tf.zeros([100]), name="bias1")
W2 = tf.Variable(tf.zeros([100, 10]), name="weights2")
b2 = tf.Variable(tf.zeros([10]), name="bias2")
Y = tf.placeholder("float", [None, 10], name="target")

# L1 = tf.matmul(X, W1) + b1
# h = tf.nn.sigmoid(L1)


def inference(X):
   return tf.nn.softmax(tf.matmul(L1, W2) + b2)

def loss(X, Y):
   return -tf.reduce_sum(Y*tf.log(inference(X)))

def train(total_loss):
   learning_rate = 0.01
   return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(X, Y):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(inference(X), 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

# dataload
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Launch the graph in session
sess=tf.Session()
sess.run(tf.global_variables_initializer())
total_loss=loss(X, Y)
train_op=train(total_loss)

# actual training steps
training_steps = 1000
for step in range(training_steps):
    batch_x, batch_y = mnist.train.next_batch(100)
    temp,loss_train=sess.run([train_op,total_loss], feed_dict={X: batch_x, Y: batch_y})
    if step%10 ==0:
        print("loss on training=", loss_train)

# Final performance Evaluation
# accuracy=sess.run(evaluate, feed_dict={X: mnist.train.images, Y: mnist.train.labels}) # why not this?
accuracy = sess.run(evaluate(mnist.train.images, mnist.train.labels))
print("accuracy in train set=", accuracy)
accuracy = sess.run(evaluate(mnist.test.images, mnist.test.labels))
print("accuracy in test set=", accuracy)

import matplotlib.pyplot as plt
import numpy as np
f, axarr = plt.subplots(2,5)
plt.set_cmap("seismic")
W_end=sess.run(W2)
for i in range(10):
    axarr[np.int(i/5),i%5].imshow(W_end[:,i].reshape([28,28]))

plt.show()

sess.close()