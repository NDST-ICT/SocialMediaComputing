import argparse
import LIS


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-dat', type=str, default=None)
    args = parser.parse_args()

    n = args.N

    p_z = [[] for _ in range(n)]
    p_o = [[] for _ in range(n)]
    LIS.load_none_dup_cascade(args.dat, p_z, p_o)

    num_po_sa = 0
    num_ne_sa = 0
    for i in range(n):
        num_po_sa = num_po_sa + len(p_o[i])
        num_ne_sa = num_ne_sa + len(p_z[i])
    print('num_po_sa:', num_po_sa, 'num_ne_sa:', num_ne_sa)

    num_po_con = 0
    num_ne_con = 0
    for i in range(n):
        for con in p_o[i]:
            num_po_con = num_po_con + con[1]
        for con in p_z[i]:
            num_ne_con = num_ne_con + con[1]
    print('num_po_con:', num_po_con, 'num_ne_con:', num_ne_con)

