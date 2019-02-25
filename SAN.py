import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#初始化,一个均匀分布，也可以用高斯分布
def xavier_init(fan_in,fan_out,constant = 1):
    low = - constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)
class AGNA(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
               optimizer = tf.train.AdamOptimizer(),scale=0.1):
#输入层节点个数
        self.n_input = n_input
#隐藏层节点个数
        self.n_hidden = n_hidden
#激活函数
        self.transfer = transfer_function
#占位符，定义训练规格
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
#定义权值，初始化
        network_weights = self._initialize_weights()
        self.weights = network_weights

#定义输入数据
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
#定义隐藏层数据,这里加了噪音，然后对W*x+b使用softplus激活函数
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x+scale * tf.random_normal((n_input,))
                                           ,self.weights['w1']),self.weights['b1']))
#复原层
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])                                    

        #损失
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))

        #训练方式，亚当梯度下降法
        self.optimizer = optimizer.minimize(self.cost)
        
        #初始化变量
        init = tf.global_variables_initializer()
        #创建会话,这里只有一个会话，所以就是默认否则需要修改
        self.sess = tf.Session()
        writer = tf.summary.FileWriter("D://logss/", self.sess.graph)
        #运行初始化
        self.sess.run(init)
    #初始化权重，w采用前面的x初始化，b采用全0初始化
    def _initialize_weights(self):
#创建一个字典
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype = tf.float32))
        return all_weights
    #定义当前损失,求取损失同时并且训练
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),
                               feed_dict={self.x:X,self.scale:self.training_scale})
        return cost
    #定义损失，不训练
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,
                               feed_dict={self.x:X,self.scale:self.training_scale})
    #定义隐藏层输出
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,
                                                    self.scale:self.training_scale})
    #定义输出层输出
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict = {self.hidden:hidden})
    #定义一个整体过程，这里包括了transform和generate
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict = {self.x:X,self.scale:self.training_scale})

    def getWeigths(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    

    #数据预处理，让数据变为0均值且标准差为1的分布
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
aec=AGNA(n_input = 784,
        n_hidden = 200,
        transfer_function = tf.nn.softplus,
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
        scale = 0.01)
X_train,X_test =standard_scale(mnist.train.images,mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1


for epoch in range(training_epochs):
    avg_cost=0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train,batch_size)

        cost = aec.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if  epoch % display_step ==0:
        print("Epoch:",'%04d'%(epoch+1),"cost=",
                  "{:.9f}".format(avg_cost))

print("Total cost:"+str(aec.calc_total_cost(X_test)))


    
                                
