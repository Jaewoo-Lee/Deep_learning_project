import tensorflow as tf
import numpy as np

# initialize variables/model parameters
W=tf.Variable(tf.zeros([2,1]), name="weight", dtype=tf.float32)
b=tf.Variable(0,name="bias", dtype=tf.float32)

# define the training loop operations
def inference(X):
    return tf.matmul(X,W)+b

def loss(X,Y):
    Y_predicted=inference(X)
    #return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
    return tf.reduce_sum(tf.square(Y-Y_predicted))

def inputs():
   weight_age=np.array([[84,73,65,70,76,69,63,72,79,75,27,89,65,57,59,69,60,79,75,82,59,67,85,55,63],
                        [46,20,52,30,57,25,28,36,57,44,24,31,52,23,60,48,34,51,50,34,46,23,37,40,30]])
   weight_age=weight_age.transpose() #weight_age=np.transpose(weight_age)
   #weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25],
   #              [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23],
   #              [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23],
   #              [85, 37], [55, 40], [63, 30]]
   blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
   return tf.to_float(weight_age), tf.to_float(blood_fat_content)

def train(total_loss):
   learning_rate = 0.01
   return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
   print(sess.run(inference([[80., 25.]])))# ~ 303
   print(sess.run(inference([[65., 25.]])))# ~ 256

# Launch the graph in session
sess=tf.Session()
sess.run(tf.global_variables_initializer())
X,Y=inputs()
total_loss=loss(X,Y)
train_op=train(total_loss)
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)

# actual training steps
training_steps = 1000
for step in range(training_steps):
  sess.run(train_op)
  if step%10 ==0:
      print("loss=", sess.run(total_loss))

evaluate(sess,X,Y)
coord.request_stop()
coord.join(threads)
sess.close()



