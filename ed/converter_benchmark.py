import numpy as np
import scipy.sparse
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default=None)
    parser.add_argument('-com', type=str, default=None)
    args = parser.parse_args()

    net_file = args.net
    com_file = args.com

    f = open(com_file, 'r')
    n = len(f.readlines())
    f.close()

    row = list()
    col = list()
    f = open(net_file, 'r')
    line = f.readline()
    while line:
        u, v = line.split()
        u = int(u)
        v = int(v)
        row.append(u - 1)
        col.append(v - 1)
        line = f.readline()
    data = list(1.0 for i in range(len(row)))
    mat = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    scipy.sparse.save_npz('net.npz', mat)
    f.close()

    f = open(com_file, 'r')
    label = list(0 for i in range(n))
    line = f.readline()
    c_num = 0
    while line:
        i, c = line.split()
        i = int(i)
        c = int(c)
        c_num = max(c_num, c)
        label[i - 1] = c
        line = f.readline()
    label = np.array(label, dtype=int)
    np.save('com.npy', label)
    f.close()

    com = [list() for _ in range(c_num)]
    for i in range(n):
        com[label[i] - 1].append(i + 1)
    f = open('file1.txt', 'w')
    for i in range(c_num):
        flag = 1
        for x in com[i]:
            if flag == 0:
                f.write(' ')
            f.write(str(x))
            flag = 0
        f.write('\n')
    f.close()

    print('net_size:', n, 'com_num:', c_num)
