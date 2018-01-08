import tensorflow as tf
import numpy as np

a = tf.Variable(3, name="my_variable")
b = tf.Variable(tf.ones([6],dtype=tf.int32))
c = tf.reduce_sum(b)
a = a+c
d = tf.add(5, a)

init = tf.global_variables_initializer() # wait! What is this???
sess = tf.Session()
sess.run(init)
print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
print(sess.run(d))
