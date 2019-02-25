def td(alpha,gamma,state_sample,action_sample,reward_sample):
    vfunc = dict()
    for s in states:
        vfunc[s] = random.random()
    for iter1 in range(len(state_sample)):
        for step in range(len(state_sample[iter1])):
            s = state_sample[iter1][step]
            r = reward_sample[iter1][step]
            if len(state_sample[iter1]) - 1 > step:
                #获取当前实验当前状态的下一状态
                s1 = state_sample[iter1][step+1]
                #获取下一状态值
                next_v = vfunc[s1]
            else:
                next_v = 0.0
               #时间差分法更新状态值 
            vfunc[s] = vfunc[s] + alpha * (r * gamma * next_v - vfunc[s])

def qlearning(num_iter1,alpha.epsilon):
    while False == t:
        key = "%d_%s"%(s,a)
        t,s1,r  = grid.transform(s,a)
        key1 = ""
        qmax = -1.0
        for a1 in actions:
            if qmax <qfunc["%d_%s"%(s1,a1)]:
                qmax = qfunc["%d_%s"%(s1,a1)]
                key1 = "%d_%s"%(s1,a1)
        qfunc[key] = qfunc[key] + alpha * ( \
            r+gamma * qfunc[key1] - qfunc[key])
        s = s1
        a = epsilon_greedy(qfunc,s1,epsilon)
        count  +=1
    return qfunc

def epsilon_greedy(qfunc,state):
    amax = 0
    key = "%d_%s"%(state,action[0])
    qmax = qfunc[key]
    #获取使得当前状态下动作行为值最大的动作
    for i in range(len(actions)):
        key = "%d_%s"%(state,action[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i

    pro= [0.0 for i in range(len(actions))]
    pro[amax] += 1 - epsilon
    for  i in range(len(actions)):
        pro[i] += epsilon/len(actions)

    r = random.random()
    s = 0.0
    for i in range(len(actions)):
        s += pro[i]
        if s >=r : return actions[i]
return actions[len(actions)-1]

def greedy(qfunc,state):
    amax = 0
    key = "%d_%s" % (state,actions[i])
    qmax = qfunc[key]
    for i in range(len(actions)):
        key = "%d_%s"%(state,actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    return actions[amax]

