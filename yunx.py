import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from SAN import AGNA
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
aec=AGNA(n_input = 784,
        n_hidden = 200,
        transfer_function = tf.nn.softplus,
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
        scale = 0.01)
X_train,X_test =aec.standard_scale(mnist.train.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1


for epoch in range(training_epochs):
    avg_cost=0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = aec.get_random_block_from_data(X_train,batch_size)

        cost = aec.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if  epoch % display_step ==0:
        print("Epoch:",'%04d'%(epoch+1),"cost=",
                  "{:.9f}".format(avg_cost))

print("Total cost:"+str(aec.calc_total_cost(X_test)))
