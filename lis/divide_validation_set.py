import argparse
import LIS
import random


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-dat', type=str, default=None)
    args = parser.parse_args()

    n = args.N
    dat_file_path = args.dat

    p_z_test = [[] for _ in range(n)]
    p_o_test = [[] for _ in range(n)]
    LIS.load_none_dup_cascade(args.dat, p_z_test, p_o_test)

    p_z_va = [[] for _ in range(n)]
    p_o_va = [[] for _ in range(n)]
    for i in range(n):
        j = 0
        while True:
            if j == len(p_z_test[i]):
                break
            r = random.randint(0, 1)
            if not r:
                p_z_va[i].append(p_z_test[i].pop(j))
            else:
                j = j + 1
        j = 0
        while True:
            if j == len(p_o_test[i]):
                break
            r = random.randint(0, 1)
            if not r:
                p_o_va[i].append(p_o_test[i].pop(j))
            else:
                j = j + 1
        print('divide_validation ', i, ' done.')

    LIS.output_dat(dat_file_path + '_test', p_z_test, p_o_test, n)
    LIS.output_dat(dat_file_path + '_validation', p_z_va, p_o_va, n)

