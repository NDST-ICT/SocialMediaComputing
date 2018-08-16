import argparse
import LIS


def load_origin_data():
    f = open(net_file, 'r')
    line = f.readline()
    while line:
        u, v = line.split()
        u = int(u)
        v = int(v)
        in_nodes[v].append(u)
        line = f.readline()
    f.close()

    f = open(cas_file, 'r')
    line = f.readline()
    while line:
        one_cas = []
        in_cas = line.split(',')
        cas_len = len(in_cas)
        if cas_len < 3:
            line = f.readline()
            continue
        i = 1
        while i < cas_len:
            one_cas.append(int(in_cas[i]))
            i = i + 2
        cas.append(one_cas)
        line = f.readline()
    f.close()


def add_p(p, con):
    found = 0
    for con_t in p:
        if con_t[0] == con:
            con_t[1] = con_t[1] + 1
            found = 1
            break
    if found == 0:
        p.append([con, 1])


def add_p_z(node, con):
    for i in range(len(con)):
        add_p(p_z[node], con[0:i+1])


def add_p_o(node, con):
    if not con:
        return
    add_p(p_o[node], con)
    for i in range(len(con)-1):
        add_p(p_z[node], con[0:i+1])


def get_p():
    cas_num = len(cas)
    for i in range(n):
        print('Constructing contexts of node ', i)
        for j in range(cas_num):
            if i == cas[j][0] or len(set(in_nodes[i]) & set(cas[j])) == 0:
                continue
            con = []
            if cas[j][0] in in_nodes[i]:
                con.append(cas[j][0])
            cas_len = len(cas[j])
            k = 1
            acd = 0
            while k < cas_len:
                if cas[j][k] == i:
                    acd = 1
                    add_p_o(i, con)
                    break
                if cas[j][k] in in_nodes[i]:
                    con.append(cas[j][k])
                k = k + 1
            if acd == 0:
                add_p_z(i, con)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-net', type=str, default=None)
    parser.add_argument('-cas', type=str, default=None)
    parser.add_argument('-out', type=str, default=None)
    args = parser.parse_args()

    n = args.N
    net_file = args.net
    cas_file = args.cas
    out_file = args.out

    in_nodes = [[] for _ in range(n)]
    cas = []
    load_origin_data()

    p_z = [[] for _ in range(n)]
    p_o = [[] for _ in range(n)]
    get_p()

    LIS.remove_duplication(p_z, n)
    LIS.remove_duplication(p_o, n)

    LIS.output_dat(out_file, p_z, p_o, n)
