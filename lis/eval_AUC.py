import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import LIS


def cal_prob(context, node):
    sum_i = np.zeros((f_len,), float)
    con_len = len(context)
    i = max(0, con_len - l - 1)
    while i < con_len:
        sum_i = np.add(sum_i, final_i[context[i]])
        i = i + 1
    prob = 1.0 - np.exp(-lmd * np.sum(np.multiply(sum_i, final_s[node])))
    return prob


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-feature', type=int, default=None)
    parser.add_argument('-l', type=int, default=None)
    parser.add_argument('-lmd', type=float, default=None)
    parser.add_argument('-dat', type=str, default=None)
    parser.add_argument('-matrix_i', type=str, default=None)
    parser.add_argument('-matrix_s', type=str, default=None)
    args = parser.parse_args()

    n = args.N
    f_len = args.feature
    l = args.l
    lmd = args.lmd
    dat_file = args.dat
    ma_i_file = args.matrix_i
    ma_s_file = args.matrix_s

    p_z = [[] for _ in range(n)]
    p_o = [[] for _ in range(n)]
    LIS.load_none_dup_cascade(dat_file, p_z, p_o)

    final_i = LIS.load_matrix(ma_i_file, n, f_len)
    final_s = LIS.load_matrix(ma_s_file, n, f_len)

    roc_true = []
    roc_score = []
    for i in range(n):
        for con_t in p_z[i]:
            prob = cal_prob(con_t[0], i)
            for _ in range(con_t[1]):
                roc_true.append(0)
                roc_score.append(prob)
        for con_t in p_o[i]:
            prob = cal_prob(con_t[0], i)
            for _ in range(con_t[1]):
                roc_true.append(1)
                roc_score.append(prob)

    roc_true = np.array(roc_true, int)
    roc_score = np.array(roc_score, float)
    auc = roc_auc_score(roc_true, roc_score)
    print('auc:', str(auc))


