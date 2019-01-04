import random
import sys
import time
import numpy as np

import argparse


V = set()
S = set()

def R_DU(point_R, point_U, v, influence_dict, point_temp):
    
    for i in point_U:
        if i in point_temp:
            continue
        influence_dict[i] -= 1
        point_R[i].remove(v)

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
        s_temp = a

    return len(temp)


def R_first_v(adj, v, point_U):
    temp = set({v})
    s_temp = set({v})
    if not(v in point_U):
        point_U[v] = set({v})
    point_U[v].add(v)
    while s_temp:
        a = set()
        for i in s_temp:
            if not(i in adj):
                continue
            for j in adj[i]:
                if not(j in temp):                
                    temp.add(j)
                    a.add(j)
                    if not(j in point_U):
                        point_U[j] = set({j})
                    point_U[j].add(v)
                        
        s_temp = a
    return len(temp), temp


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=None)
    parser.add_argument('-dat', type=str, default=None)
    args = parser.parse_args()

    K = args.k
    graph_file = args.dat

    t_start = time.time()
    f = open(graph_file,"r")
    adj={}
    G = []
    cnt = 0
    sta = {}
    point_R = []
    point_U = []
    for i in range(100):
        G.append({})
        point_R.append({})
        point_U.append({})
    fr = f.readlines()
    for line in fr[1:]:
        a = line.split('\r')[0].split(' ')
        b = int(a[0])
        c = int(a[1])
        if not(c in sta):
            sta[c] = 0.0
        sta[c] += 1.0
        if not(b in sta):
            sta[b] = 0.0
        sta[b] += 1.0  
        #Count Occurrences
    for line in fr[1:]:
        a = line.split('\r')[0].split(' ')
        b = int(a[0])
        c = int(a[1])
        if not(b in adj):
            adj[b] = []    
        adj[b].append(c)
        if not(c in adj):
            adj[c] = []    
        adj[c].append(b)
        for j in range(100):
            if random.random() < (1.0 / sta[c]):
                if not(b in G[j]):
                   G[j][b] = []
                G[j][b].append(c)
            if random.random() < (1.0 / sta[b]):
                if not(c in G[j]):
                   G[j][c] = []
                G[j][c].append(b)
    t_adj = time.time()
    print("generating adj time: ", t_adj - t_start)
    V = set(adj.keys())
    print("Finishing generating Gi!")
    all_num = int(fr[0].split(" ")[1])
    print(all_num)
    print("G:", sys.getsizeof(G))

    influence_dict = {}

    for i in V:
        num = 0 
        for j in range(100):
            a, point_R[j][i] = R_first_v(G[j], i, point_U[j])
            num += a
        influence_dict[i] = num

    point = sorted(influence_dict, key = lambda a: influence_dict[a], reverse=True)[0]
    
    print("calculating time: ", time.time() - t_adj)
    S.add(point)
    print("step: ", 0, "num", influence_dict[point], "S", S)

    for i in range(K-1):
        for j in range(100):
            if point in point_R[j]:
                point_temp = point_R[j][point].copy()
                for k in point_temp:
                    influence_dict[k] -= len(point_R[j][k])
                    point_R[j][k].clear()
                for k in point_temp:
                    R_DU(point_R[j], point_R[j][k], k, influence_dict, point_temp)

        point = sorted(influence_dict, key = lambda a: influence_dict[a], reverse=True)[0]
        S.add(point)

        print("step: ", i + 1, "num", influence_dict[point], "S", S)
        influence_dict[point] = 0

    t_finds = time.time()
    t = t_finds - t_start
    print("Time: ", t)
    print(S)
    
    cnt = 0
    for i in range(10000):
        cnt += cal_R(adj, S, sta)
    cnt /= 10000
    #verify the result
    print("influence: ", cnt)
    print("cal_time: ", time.time() - t_finds)








