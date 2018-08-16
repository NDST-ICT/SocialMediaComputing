from sklearn.decomposition import NMF
import scipy.sparse
import numpy as np
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=None)
    args = parser.parse_args()

    A = scipy.sparse.load_npz('net.npz')
    n = A.shape[0]
    k = args.k

    nmf_model = NMF(n_components=k)
    W = nmf_model.fit_transform(A)
    Z = nmf_model.components_
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
