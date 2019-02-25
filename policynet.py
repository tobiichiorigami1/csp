import numpy as np
import tensorflow as tf
import gym
env = gym.make('CartPole-v0')
env.reset()
#隐含层节点数
H = 50
batch_size = 25
learning_rate = 0.1
D = 4
gamma = 0.99
#定义输入，输入形状为4
observations = tf.placeholder(tf.float32,[None,D],name ="input_x")
#第一层权重
w1 = tf.get_variable(name="W1",shape=[D,H],
                    initializer = tf.contrib.layers.xavier_initializer())
#第一层为全连接，使用relu激活i
layer1 = tf.nn.relu(tf.matmul(observations,w1))
#第二层权重
w2 = tf.get_variable(name="W2",shape=[H,1],
                     initializer = tf.contrib.layers.xavier_initializer())
#输出层激活函数为sigmoid
score = tf.matmul(layer1,w2)
probability = tf.nn.sigmoid(score)
adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
W1Grad = tf.placeholder(tf.float32,name = "batch_grad1")
W2Grad = tf.placeholder(tf.float32,name = "batch_grad2")
batchGrad = [W1Grad,W2Grad]
#获取所有可训练的参数，只有w1,w2
tvars = tf.trainable_variables()
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))
#估算每个动作的对应潜在价值
def discount_reward(r):
    discounted_r = np.zeros_like(r)
    #累计回报
    running_add = 0
    #从后向前进行各个动作的值计算
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
input_y = tf.placeholder(tf.float32,[None,1],name = "input_y")
advantages = tf.placeholder(tf.float32,name = "reward_signal")
#这里的似然估计，很难想到，首先分别列出标签为1，0的且判断正确的概率，
#这里标签为0，action为1的概率为probability，反之为input_y - probability，
#然后将这两项相加去配使得带入标签值后同时满足上面值的式子，
#最终得到下面的式子，然后log一下后，这个值越小说明似然估计越大
loglik = tf.log(input_y*(input_y - probability) + \
                (1 - input_y)*(input_y + probability))
#这里损失函数在上式再乘以一个潜在价值，取相反数
loss = -tf.reduce_mean(loglik * advantages)

#求取训练参数对于损失函数的梯度
newGrads = tf.gradients(loss,tvars)
#定义环境，动作，回报列表
xs,ys,drs = [],[],[]
#累计回报
reward_sum = 0
#迭代计数器
episode_number = 1
#最大迭代次数
total_episodes = 2000
#创建会话开始进行图计算:
with tf.Session() as sess:
    rendering  =  False
    init = tf.global_variables_initializer()
    sess.run(init)

    observation = env.reset()

    gradBuffer = sess.run(tvars)

    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
#开始训练
    while episode_number <= total_episodes:

        if reward_sum/batch_size > 100 or rendering == True :
            env.render()
            rendering = True


        x = np.reshape(observation,[1,D])


        tfprob = sess.run(probability,feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)

        y = 1 - action
        ys.append(y)

        observation,reward,done,info = env.step(action)

        reward_sum += reward

        drs.append(reward)

        if done :
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)

            xs,ys,drs = [],[],[]
            #标准化潜在价值:减去均值再除以标准差
            discounted_epr = discount_reward(epr)
            discounted_epr -=  np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            tGrad = sess.run(newGrads,feed_dict={observations: epx,
                                                 input_y:epy,advantages:discounted_epr})

            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            #每训练25次进行一次梯度的更新
            if episode_number % batch_size == 0:
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],
                                                W2Grad:gradBuffer[1]})
            #清空梯度
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] *= 0

            #输出当前的平均累积回报
                print('%d,%s'%(episode_number,reward_sum/batch_size))

            #如果平均累积回报大于200,停在训练
                #if reward_sum/batch_size > 200:
                    #break


                #累积下25次的回报
                reward_sum = 0
                
            observation = env.reset()

            
                    

                

        
        
        
    


    





