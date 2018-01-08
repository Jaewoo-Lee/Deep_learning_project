import tensorflow as tf
import numpy as np

# Create variable with starting value of 1
my_var = tf.Variable(1)
# Create an operation that multiplies the variable by 2 each time it is run
my_var_times_two = my_var.assign(my_var * 2)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(my_var))
print(sess.run(my_var_times_two))
print(sess.run(my_var_times_two))
print(sess.run(my_var.assign_add(1)))
print(sess.run(my_var.assign_sub(2)))
