import numpy as np
import math
import argparse
import json


def cal_NMI(C1, C2):
    #This is NMI for partitions.
    C1 = list(C1)
    C2 = list(C2)
    length = len(C1)
    set_C1 = set(C1)
    set_C2 = set(C2)

    prox = dict()
    for c in set_C1:
        prox[c] = 1.0*C1.count(c)/length

    proy = dict()
    for c in set_C2:
        proy[c] = 1.0*C2.count(c)/length

    C_x = list()
    for i in range(length):
        C_x.append((C1[i], C2[i]))

    I=0.0
    for c1 in set_C1:
        for c2 in set_C2:
            proxy = 1.0*C_x.count((c1, c2))/length
            if proxy > 0:
                I = I+proxy*math.log(proxy/(prox[c1]*proy[c2]), 2)

    H_x=0.0
    H_y=0.0
    for c1 in set_C1:
        H_x = H_x-prox[c1]*math.log(prox[c1],2)
    for c2 in set_C2:
        H_y = H_y-proy[c2]*math.log(proy[c2],2)

    NMI = 2.0*I/( H_x+H_y )
    return NMI


def cal_purity(C, L):
    C = list(C)
    c_size = list(0 for _ in range(k))
    for c in C:
        c_size[c] = c_size[c] + 1

    c_max = list(0 for _ in range(k))
    for i in range(len(L)):
        table_cij = dict()
        for x in L[i]:
            if C[x-1] not in table_cij:
                table_cij[C[x-1]] = 1
            else:
                table_cij[C[x-1]] = table_cij[C[x-1]] + 1
            c_max[C[x-1]] = max(c_max[C[x-1]], table_cij[C[x-1]])

    P = 0.0
    for i in range(k):
        if c_size[i] > 0:
            P = P + float(c_max[i]) / c_size[i]
    P = P / k
    return P


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default=None)
    parser.add_argument('-k', type=int, default=None)
    args = parser.parse_args()

    k = args.k
    out = args.c
    pre = np.load('results.npy')

    if out == 'NMI':
        std_com = np.load('com.npy')
        print('NMI:', cal_NMI(pre, std_com))
    if out == 'purity':
        with open('label.json', 'r') as f:
            label = json.load(f)
        print('purity:', cal_purity(pre, label))
