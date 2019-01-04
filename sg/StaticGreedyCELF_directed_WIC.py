import random
import sys
import time
import numpy as np
from collections import deque
import argparse

V = set()
S = set()


def R_v(adj, v, former_temp):
    temp = set()
    if not(v in former_temp):
        temp.add(v)
    s_temp = set({v})
    while s_temp:
        a = set()
        for i in s_temp:
            if not(i in adj):
                continue
            for j in adj[i]:
                if not(j in temp) and not(j in former_temp):                
                    temp.add(j)
                    a.add(j)
        s_temp = a
    return len(temp)

def R_first_v(adj, v):
    temp = set({v})
    s_temp = set({v})
    while s_temp:
        a = set()
        for i in s_temp:
            if not(i in adj):
                continue
            for j in adj[i]:
                if not(j in temp):                
                    temp.add(j)
                    a.add(j)
        s_temp = a
    return len(temp)

def cal_R(adj, S, sta):
    temp = set()
    s_temp = S.copy()
    temp = temp | s_temp
    while s_temp:
        a = set()
        for i in s_temp:
            if not(i in adj):
                continue
            for j in adj[i]:
                b = random.random()
                if b < (1.0 / sta[j]) and not(j in temp):
                    temp.add(j)
                    a.add(j)
        #s_temp = a.copy()
        s_temp = a

    return len(temp)

def R_S(adj, v, former_temp):
    s_temp = set()
    if v == 1000000:
        return 0
    s_temp.add(v)
    former_temp.add(v)
    while s_temp:
        a = set()
        for i in s_temp:
            if not(i in adj):
                continue
            for j in adj[i]:
                if not(j in former_temp):                
                    former_temp.add(j)
                    a.add(j)
        s_temp = a


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=None)
    parser.add_argument('-dat', type=str, default=None)
    args = parser.parse_args()

    K = args.k
    graph_file = args.dat

    t_start = time.time()
    f = open(graph_file,"r")
    adj = {}
    G = []
    cnt = 0
    sta = {}
    for i in range(100):
        G.append({})
    fr = f.readlines()
    for line in fr[4:]:
        a = line.split('\r')[0].split('\t')
        if not(int(a[1]) in sta):
            sta[int(a[1])] = 0.0
        sta[int(a[1])] += 1.0
    for line in fr[4:]:
        a = line.split('\r')[0].split('\t')
        b = int(a[0])
        c = int(a[1])
        if not(b in adj):
            adj[b] = []    
        adj[b].append(c)
        for j in range(100):
            if random.random() < (1.0 / sta[c]):
                if not(b in G[j]):
                   G[j][b] = []
                G[j][b].append(c)
    t_adj = time.time()
    print("generating adj time: ", t_adj - t_start)
    V = set(adj.keys())
    print("Finishing generating Gi!")

    influence_dict = {}

    for i in V:
        num = 0 
        for j in G:
            num += R_first_v(j, i)
        influence_dict[i] = num
    print("cal_time: ", time.time() - t_adj)
    order = sorted(influence_dict, key = lambda a: influence_dict[a], reverse=True)
    former_temp = []
    for i in range(len(G)):
        former_temp.append(set())
    S.add(order[0])
    print("step: ", 0, "num", influence_dict[order[0]], "S", S)
    for j in range(len(G)):
        R_S(G[j], order[0], former_temp[j])
    influence_dict.pop(order[0])
    order.remove(order[0])
    for i in range(K-1):
        k = 0
        index_ = 0
        maximum = 0
        while 1:
            v = order[k]
            num = 0
            for j in range(len(G)):
                num += R_v(G[j], v, former_temp[j])
            
            influence_dict[v] = num
            if num > maximum:
                maximum = num
                index = k
            influence_dict[v] = num
            if influence_dict[order[k]] < influence_dict[order[k + 1]]:
                k += 1
            else:
                break
            
            if maximum > influence_dict[order[k + 1]]:
                k = index
                break
        S.add(order[k])
        for j in range(len(G)):
            R_S(G[j], order[k], former_temp[j])
        
        print("step: ", i + 1, "num", influence_dict[order[k]], "S", S)
        influence_dict.pop(order[k])
        order = sorted(influence_dict, key = lambda a: influence_dict[a], reverse=True)

    t_finds = time.time()
    t = t_finds - t_start
    print("Time: ", t)
    print(S)
    
    cnt = 0
    for i in range(10000):
        cnt += cal_R(adj, S, sta)
    cnt /= 10000
    
    print("influence: ", cnt)
    print("cal_time: ", time.time() - t_finds)
        
        
        
            
    
    
    






















