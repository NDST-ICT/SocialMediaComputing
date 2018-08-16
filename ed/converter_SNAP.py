import argparse
import scipy.sparse
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default=None)
    parser.add_argument('-label', type=str, default=None)
    args = parser.parse_args()

    net_file = args.net
    label_file = args.label

    n = 0
    table = dict()
    row = list()
    col = list()
    f = open(net_file, 'r')
    line = f.readline()
    while line:
        if line[0] == '#':
            line = f.readline()
            continue
        u, v = line.split()
        u = int(u)
        v = int(v)
        if u not in table:
            n = n + 1
            table[u] = n
        if v not in table:
            n = n + 1
            table[v] = n
        row.append(table[u] - 1)
        col.append(table[v] - 1)
        row.append(table[v] - 1)
        col.append(table[u] - 1)
        line = f.readline()
    data = list(1.0 for i in range(len(row)))
    mat = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    scipy.sparse.save_npz('net.npz', mat)
    f.close()

    label = list()
    f = open(label_file, 'r')
    line = f.readline()
    while line:
        nodes = line.split()
        group = list()
        for x in nodes:
            group.append(table[int(x)])
        label.append(group)
        line = f.readline()
    f.close()

    with open('label.json', 'w') as f:
        json.dump(label, f)
    print('net_size:', n, 'label_num:', len(label))
