import numpy as np
import scipy.sparse
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=None)
    args = parser.parse_args()

    A = scipy.sparse.load_npz('net.npz')
    n = A.shape[0]
    k = args.k

    W = np.random.rand(n, k)
    Z = np.random.rand(k, n)
    epsilon = 0.0

    cou = 0
    while cou < 500:
        up_w = A * Z.T
        down_w = np.dot(W, np.dot(Z, Z.T))
        up_z = W.T * A
        down_z = np.dot(np.dot(W.T, W), Z)
        W = epsilon * W + (1 - epsilon) * np.multiply(W, up_w / np.maximum(down_w, 1e-80))
        W = np.where(W > 1e-80, W, 0.0)
        Z = epsilon * Z + (1 - epsilon) * np.multiply(Z, up_z / np.maximum(down_z, 1e-80))
        Z = np.where(Z > 1e-80, Z, 0.0)
        cou = cou + 1
        if cou % 100 == 1:
            epsilon = epsilon + 0.1999
        print("ite_count:", cou)

    pre = np.argmax(W, 1)
    np.save('results.npy', pre)

    label = list(pre)
    com = [list() for _ in range(k)]
    for i in range(n):
        com[label[i]].append(i + 1)
    f = open('file2.txt', 'w')
    for i in range(k):
        flag = 1
        for x in com[i]:
            if flag == 0:
                f.write(' ')
            f.write(str(x))
            flag = 0
        f.write('\n')
    f.close()
