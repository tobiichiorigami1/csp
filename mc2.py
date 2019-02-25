def mc(num_iter1,epsilon):
    x = []
    y = []
    n = dict(0)
    qfunc = dict()
#初始化动作状态值为0
    for s in states:
        for a in actions:
            qfunc["%d_%s"%(s,a)] = 0.0
            n["%d_%s"%(s,a)] = 0.001
    for iter1 in range(num_iter1):
        x.append(iter1)
        y.append(compute_error(qfunc))
        s_sample = []
        a_sample = []
        r_sample = []
        s = states[int(random.random()*len(states))]
        t = False
        count =0
        while False ==t and count < 100:
            a =epsilon_greedy(qfunc,s,epsilon)
            t,s1,r = grid.transform(s,a)
            s_sample.append(s)
            a_sample.append(a)
            r_sample.append(r)
            s=s1
            count+=1
        g = 0.0
        for i in range(len(s_sample)-1,-1,-1):
            g *= gamma
            g+= r_sample[i]
        for i  in range(len(s_sample)):
            key = "%d_%s"%(s_sample[i],a_sample[i])
            n[key] += 1.0
            qfunc[key] = (qfunc[key]*(n[key]-1)+g)/n[key]
            g -= r_sample[i]
            g /= gamma
        return qfunc
    
                            
            
        
