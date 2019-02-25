from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os



mnist = input_data.read_data_sets("MINIST_data/",one_hot=True)

save_dir='D://myimage/'

if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range(20):
    #image_array = mnist.train.images[i,:]

    #image_array = image_array.reshape(28,28)

    #filename = save_dir + '%d' %i+'.jpg'

    #scipy.misc.imsave( filename, image_array)
    print(mnist.train.labels[i,:])
