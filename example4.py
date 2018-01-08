import tensorflow as tf
import numpy as np
a = tf.placeholder(tf.int32, shape=[2], name="my_input") # must have datatype

# Use the placeholder as if it were any other Tensor object
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b, c, name="add_d")

#Create a dictionary to pass into 'feed_dict'
#Key: 'a', the handle to the placeholder's output Tensor
#Value: A vector with value [5, 3] and int32 data type
input_dict = {a: np.array([5, 3], dtype=np.int32)}

# Fetch the value of 'd', feeding the values of 'input_vector' into 'a'
sess = tf.Session()
print(sess.run(d, feed_dict=input_dict))
