import RPP_prior
import math
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-T', type=float, default=None)
    parser.add_argument('-m', type=float, default=None)
    parser.add_argument('-ET', type=int, default=None)
    parser.add_argument('-dat', type=str, default=None)
    parser.add_argument('-itv', type=float, default=None)
    parser.add_argument('-out', type=str, default=None)
    args = parser.parse_args()

    time_l = []
    file = open(args.dat, 'r')
    for _ in range(args.N):
        line = file.readline().split('\t')
        td = []
        for t in line:
            t = float(t)
            td.append(t)
        time_l.append(td)
    file.close()

    mu = []
    sigma = []
    nd = []
    X = []
    file = open(args.out, 'r')
    line = file.readline().split('\t')
    alpha = float(line[0])
    beta = float(line[1])
    for _ in range(args.N):
        line = file.readline().split('\t')
        mu.append(float(line[0]))
        sigma.append(float(line[1]))
        nd.append(float(line[2]))
        X.append(float(line[3]))
    file.close()

    t = args.itv
    re = open('eval_result_prior.txt', 'w')
    while t <= args.ET:
        ac = 0
        mape = 0
        print 1
        for i in range(args.N):
            rt = 0
            for x in time_l[i]:
                if x <= args.T + t:
                    rt = rt + 1
            ct = RPP_prior.predict(args.m, nd[i], args.T, alpha, beta, mu[i], sigma[i], X[i], args.T + t)
            mape = mape + math.fabs(ct - rt) / rt
            if math.fabs(ct - rt) / rt <= 0.1:
                ac = ac + 1
        mape = mape / args.N
        ac = ac / args.N
        re.write(str(mape) + '\t' + str(ac) + '\n')
        t = t + args.itv
    re.close()

