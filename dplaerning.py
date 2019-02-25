import numpy as np
import tensorflow as tf
import gym
#隐藏层节点数
H = 50
#一次训练的个数
batch_size = 25
#学习速率
learning_rate = 1e-1
#环境信息维度（小车位置，速度，杆的角度，速度）
D = 4
#折扣因子
gamma = 0.99
#输入
observations = tf.placeholder(tf.float32,[None,D],name="input_x")
#第一层权重
W1 = tf.get_variable("W1",shape=[D,H],
                     initializer=tf.contrib.layers.xavier_initializer())
#第一层输出
layer1 = tf.nn.relu(tf.matmul(observations,W1))
#第二层权重
W2 = tf.get_variable("W2",shape = [H,1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
#输出
probability = tf.nn.sigmoid(score)
#训练方法
adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
#梯度
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name = "batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))
def discount_rewards(r):
#初始化潜在价值为0，形状为r，为每一个动作所获得的实际回报
    discounted_r = np.zeros_like(r)
    running_add = 0
#从后向前求取累积潜在价值
    for t in reversed(range(r.size)):
        #从后累积价值
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
#虚拟概率
input_y = tf.placeholder(tf.float32,[None,1],name="input_y")
#潜在价值
advantages = tf.placeholder(tf.float32,name="reward_signal")
#交叉熵损失函数
loglik = tf.log(input_y*(input_y - probability) + \
                (1-input_y)*(input_y + probability))

loss = -tf.reduce_mean(loglik * advantages)
#获取全部可训练的参数
tvars = tf.trainable_variables()
#获取梯度值
newGrads = tf.gradients(loss,tvars)

xs,ys,drs =[],[],[]
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        if reward_sum/batch_size > 100 or rendering == True :
            env.render()
            rendering  = True

        x = np.reshape(observation,[1,D])

        tfprob = sess.run(probability,feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x)
        y = 1 - action
        ys.append(y)

        observation,reward,done,info = env.step(action)

        reward_sum +=reward

        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs,ys,drs = [],[],[]
            #求取每一步动作的潜在价值
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            tGrad = sess.run(newGrads,feed_dict={observations: epx,
                                                 input_y:epy,advantages:discounted_epr})

            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            if episode_number % batch_size == 0:
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],
                                                W2Grad: gradBuffer[1]})

                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                print('Average reward for episode %d : %f.'% \
                      (episode_number,reward_sum/batch_size))

                if reward_sum/batch_size > 200:
                   print("task solved in",episode_number,'episodes!')
                   break

                reward_sum =0

            observation = env.reset()
        
        


