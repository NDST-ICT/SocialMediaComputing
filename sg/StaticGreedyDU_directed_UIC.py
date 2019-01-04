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

def cal_R(adj, S):
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
                if b < 0.01 and not(j in temp):
                    temp.add(j)
                    a.add(j)
        s_temp = a.copy()

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


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=None)
    parser.add_argument('-dat', type=str, default=None)
    args = parser.parse_args()

    K = args.k
    graph_file = args.dat


    t_start = time.time()
    f = open(graph_file,"r")
    cnt = 0
    point_R = []
    point_U = []
    fr = f.readlines()
    all_num = int(fr[2].split(" ")[2])
    for i in range(100):
        point_R.append({})
        point_U.append({})
    for line in fr[4:]:
        a = line.split('\r')[0].split('\t')
        b = int(a[0])
        c = int(a[1])
        for j in range(100):
            if random.random() < 0.01:
                R = point_R[j]
                U = point_U[j]
                if not(b in point_R[j]):
                    point_R[j][b] = set({b})
                if not(b in point_U[j]):
                    point_U[j][b] = set({b})
                if not(c in point_R[j]):
                    point_R[j][c] = set({c})
                if not(c in point_U[j]):
                    point_U[j][c] = set({c})
                if not(b in U[c]):
                    for k in U[b]:
                        R[k] |= R[c]
                    for k in R[c]:
                        U[k] |= U[b]
    t_adj = time.time()
    print("generating adj time: ", t_adj - t_start)
    influence_dict = {}

    for i in range(all_num):
        num = 0 
        for j in range(100):
            if i in point_R[j]:
                num += len(point_R[j][i])
            else:
                num += 1
        influence_dict[i] = num

    point = sorted(influence_dict, key = lambda a: influence_dict[a], reverse=True)[0]
    print(influence_dict[point])

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
                    R_DU(point_R[j], point_U[j][k], k, influence_dict, point_temp)

        point = sorted(influence_dict, key = lambda a: influence_dict[a], reverse=True)[0]
        S.add(point)
        print("step: ", i + 1, "num", influence_dict[point], "S", S)
        influence_dict[point] = 0

    t_finds = time.time()
    t = t_finds - t_start
    print("Time: ", t)
    print(S)
    adj = {}
    for line in fr[4:]:
        a = line.split('\r')[0].split('\t')
        if not(int(a[0]) in adj):
            adj[int(a[0])] = []    
        adj[int(a[0])].append(int(a[1]))

    cnt = 0
    for i in range(10000):
        cnt += cal_R(adj, S)
    cnt /= 10000
    result.append(cnt)

    print("influence: ", cnt)
    print("cal_time: ", time.time() - t_finds)
    








