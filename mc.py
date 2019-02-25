def gen_randompi_sample(self,num):
    state_sample = [] #状态采样
    action_sample = [] #动作采样
    reward_sample = [] #回报采样
#开始试验
    for i in range(num):
        s_tmp = []
        a_tmp = []
        r_tmp = []
        #随机初始化状态
        s = seolf.actions[int(random.random()*len(self.states))]
        t = False
        while False == t:
            #随机选取动作
            a = self.actions[int(random.random()*len(self.actions))]
            #获取当前状态执行该动作的下一个状态与执行当前动作回报，并判断当前状态是否为终止状态
            t,s1,r = self.transform(s,a)
            #保存当前序列
            s_tmp.append(s)
            a_tmp.append(a)
            r_tmp.append(r)
            #转到下一个状态
            s=s1
            #将当前序列保存
        state_sample.append(s_tmp)
        reward_sample.append(r_tmp)
        action_sample.append(a_tmp)
#返回样本集
    return state_sample,action_sample,reward_sample

def mc(gamma,state_sample,action_sample,reward_sample):
    #各个状态的状态值
    vfunc = dict()
    #各个状态的出现次数
    nfunc = dict()
    for s in states:
        vfunc[s] = 0.0
        nfunc[s] = 0.0
        #对每次试验有
    for iter1 in range(len(state_sample)):
        G = 0.0
        #反向计算累积回报值
        for step in range(len(state_sample[iter1])-1,-1,-1):
            G *= gamma
            G += reward_sample[iter1][step]
           #这里正向计算值函数的累加 
        for step in range(len(state_sample)):
            #获取状态,从初始状态开始一直到终止状态
            s = state_sample[iter1][step]
            #累加状态值
            V[s] += G
            #状态次数计数器
            N[s] += 1
            #每求完一个状态，其回报应该减少当前状态的回报
            G -= reward_sample[iter1][step]
            #同时除以折扣因子
            G /= gamma

        #求取平均值
    for s in states:
        if N[s] >=0.0000001:
            V[s] /= N[s]
    return V
            
            
