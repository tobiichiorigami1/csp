import gym
import time
import sys
import random
import numpy as np

class GridMdp:
    def __init__(s):
        s.gamma = 0.9
        s.states = range(1,26) #状态空间
        s.actions = ['n','e','s','w'] #动作空间
        s.terminate_states = {15:1.0, 4:-1.0,9:-1.0,\
                              11:-1.0.12:-1.0,23:-1.0,24:-1.0,25:-1.0}#结束状态
        s.trans={}#转移矩阵
        for state in s.state:
            if not state in s.terminate_states:
                s.trans[state] ={}
        s.trans[1]['e'] = 2
        s.trans[1]['s'] = 6
        s.trans[2]['e'] = 3 
        s.trans[2]['w'] = 1
        s.trans[2]['s'] = 7
        s.trans[3]['e'] = 4
        s.trans[3]['w'] = 2
        s.trans[3]['s'] = 8
        s.trans[5]['w'] = 4
        s.trans[5]['s'] = 10
        s.trans[6]['e'] = 7
        s.trans[6]['s'] = 11
        s.trans[6]['n'] = 1
        s.trans[7]['e'] = 8
        s.trans[7]['w'] = 6 
        s.trans[7]['s'] = 12
        s.trans[7]['n'] = 2
        s.trans[8]['e'] = 9
        s.trans[8]['w'] = 7 
        s.trans[8]['s'] = 13
        s.trans[8]['n'] = 3
        s.trans[10]['w'] = 9
        s.trans[10]['s'] = 15
        s.trans[13]['e'] = 14
        s.trans[13]['w'] = 12 
        s.trans[13]['s'] = 18
        s.trans[13]['n'] = 8
        s.trans[14]['e'] = 15
        s.trans[14]['w'] = 13
        s.trans[14]['s'] = 19
        s.trans[14]['n'] = 9
        s.trans[16]['e'] = 17
        s.trans[16]['s'] = 21
        s.trans[16]['n'] = 11
        s.trans[17]['e'] = 18
        s.trans[17]['w'] = 16 
        s.trans[17]['s'] = 22
        s.trans[17]['n'] = 12
        s.trans[18]['e'] = 19
        s.trans[18]['w'] = 17 
        s.trans[18]['s'] = 23
        s.trans[18]['n'] = 13
        s.trans[19]['e'] = 20
        s.trans[19]['w'] = 18 
        s.trans[19]['s'] = 24
        s.trans[19]['n'] = 14
        s.trans[20]['w'] = 19
        s.trans[20]['s'] = 25
        s.trans[20]['n'] = 15
        s.trans[21]['e'] = 22
        s.trans[21]['n'] = 16
        s.trans[22]['e'] = 23
        s.trans[22]['w'] = 21
        s.trans[22]['n'] = 17

        s.rewards = {} #奖励
        for state in s.states:
            s.rewards[state] = {}
            for action in s.actions:
                s.rewards[state][action] = 0
                if state in s.trans and action in s.trans[state]:
                    next_state = s.trans[state][action]
                    if next_state in s.terminate_states:
                        s.rewards[state][action] = s.terminate_states[next_state]
        s.pi = {} #策略
        for state in s.trans:
            s.pi[state] = random.choice(s.trans[state].keys())
        s.last_pi = s.pi.copy()
        
