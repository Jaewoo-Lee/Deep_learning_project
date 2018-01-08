import tensorflow as tf
with tf.name_scope("Scope_A"): # note the colon
   a = tf.add(1, 2, name="A_add")
   b = tf.multiply(a, 3, name="A_mul")

with tf.name_scope("Scope_B"):   # note the colon
   c = tf.add(4, 5, name="B_add")
   d = tf.multiply(c, 6, name="B_mul")

e = tf.add(b, d, name="output")

writer = tf.summary.FileWriter('./ch3_ns1', tf.get_default_graph())
writer.close()
