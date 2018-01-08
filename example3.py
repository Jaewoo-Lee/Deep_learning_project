import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = a*b
d = a+b
e = c+d

sess=tf.Session()
writer=tf.summary.FileWriter('ch3_ex3', sess.graph)
print(sess.run(a))
writer.close()
sess.close()
