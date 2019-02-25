import tensorflow as tf
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
#v=v+1
sess=tf.Session()
sess.run(tf.global_variables_initializer())
with tf.control_dependencies([assignment]):
  w = v.read_value()
  print(sess.run(w))
  print(sess.run(v))
