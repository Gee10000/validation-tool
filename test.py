#import TensorFlow library
import tensorflow as tf
a=tf.constant(5)
b=tf.Variable(3)
c=tf.add(a,b)
print(c)