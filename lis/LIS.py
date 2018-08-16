import tensorflow as tf
import numpy as np
import argparse
import random
import time


'''
def load_processed_cascade(dat_file, p_z, p_o):
    f = open(dat_file, 'r')
    line = f.readline()
    while line:
        info = line.split(',')
        node = int(info[0])
        i = 1
        info_len = len(info)
        while i < info_len:
            kind, con, num = info[i].split('|')
            con = con.split('-')
            con_len = len(con)
            for j in range(con_len):
                if con[j][-1] == '*':
                    con[j] = con[j][:-1]
                con[j] = int(con[j])
            num = int(num)
            if kind == '0':
                j = con_len
                while j > 0:
                    p_z[node].append([con[0:j], num])
                    j = j - 1
            if kind == '1':
                p_o[node].append([con, num])
                j = con_len - 1
                while j > 0:
                    p_z[node].append([con[0:j], num])
                    j = j - 1
            i = i + 1
        line = f.readline()
    f.close()
'''


def load_none_dup_cascade(dat_file, p_z, p_o):
    f = open(dat_file, 'r')
    line = f.readline()
    while line:
        info = line.split(',')
        node = int(info[0])
        i = 1
        info_len = len(info)
        while i < info_len:
            kind, con, num = info[i].split('|')
            con = con.split('-')
            if not con:
                i = i + 1
                continue
            con_len = len(con)
            for j in range(con_len):
                if con[j][-1] == '*':
                    con[j] = con[j][:-1]
                con[j] = int(con[j])
            num = int(num)
            if kind == '0':
                p_z[node].append([con, num])
            if kind == '1':
                p_o[node].append([con, num])
            i = i + 1
        line = f.readline()
    f.close()


def load_matrix(path, n, m):
    ma = []
    ma_f = open(path, 'r')
    for i in range(n):
        line = ma_f.readline().split('\t')
        for j in range(m):
            ma.append(float(line[j]))
    ma_f.close()
    ma = np.array(ma)
    return ma.reshape((n, m))


def remove_duplication(p_s, n):
    for i in range(n):
        psi = p_s[i]
        j = 0
        while 1:
            if j == len(psi):
                break
            k = j + 1
            while 1:
                if k == len(psi):
                    break
                if psi[k][0] == psi[j][0]:
                    psi[j][1] = psi[j][1] + psi[k][1]
                    psi.pop(k)
                else:
                    k = k + 1
            j = j + 1
        print('remove_duplication ', i, ' done.')


def output_dat(path, p_z, p_o, n):
    out_dat_f = open(path, 'w')
    for i in range(n):
        line = str(i)
        for con_t in p_z[i]:
            if not con_t[0]:
                continue
            line = line + ',0|'
            for pr in con_t[0]:
                line = line + str(pr) + '-'
            line = line[:-1] + '|' + str(con_t[1])
        for con_t in p_o[i]:
            if not con_t[0]:
                continue
            line = line + ',1|'
            for pr in con_t[0]:
                line = line + str(pr) + '-'
            line = line[:-1] + '|' + str(con_t[1])
        if line == str(i):
            continue
        line = line + '\n'
        out_dat_f.write(line)
    out_dat_f.close()


def get_batch():
    po_pr = []
    po_af = []
    po_num = []
    for i in range(po_batch_size):
        index = random.randint(0, n-1)
        while not p_o[index]:
            index = random.randint(0, n-1)
        po_af.append(index)
        con_t = random.choice(p_o[index])
        con_len = len(con_t[0])
        j = max(0, con_len - l - 1)
        while j < con_len:
            po_pr.append([i, con_t[0][j]])
            j = j + 1
        po_num.append(con_t[1])

    ne_pr = []
    ne_af = []
    ne_num = []
    for i in range(ne_batch_size):
        index = random.randint(0, n-1)
        while not p_z[index]:
            index = random.randint(0, n-1)
        ne_af.append(index)
        con_t = random.choice(p_z[index])
        con_len = len(con_t[0])
        j = max(0, con_len - l - 1)
        while j < con_len:
            ne_pr.append([i, con_t[0][j]])
            j = j + 1
        ne_num.append(con_t[1])

    po_pr = np.array(po_pr, int)
    po_pr_v = np.ones((po_pr.shape[0], ), float)
    po_af = np.array(po_af, int)
    po_num = np.array(po_num, float)
    ne_pr = np.array(ne_pr, int)
    ne_pr_v = np.ones((ne_pr.shape[0], ), float)
    ne_af = np.array(ne_af, int)
    ne_num = np.array(ne_num, float)
    return po_pr, po_pr_v, po_af, po_num, ne_pr, ne_pr_v, ne_af, ne_num


def output_matrix(path, ma, n, m):
    out_ma_f = open(path, 'w')
    for i in range(n):
        for j in range(m):
            out_ma_f.write(str(ma[i][j]) + '\t')
        out_ma_f.write('\n')
    out_ma_f.close()


class LIS:
    def __init__(self, n, f_len, lmd, lr):
        self.n = n
        self.f_len = f_len
        self.lmd = lmd
        self.lr = lr

        self.initi = tf.cast(tf.random_uniform([n, f_len], minval=0.0, maxval=1.0), tf.float64)
        self.inits = tf.cast(tf.random_uniform([n, f_len], minval=0.0, maxval=1.0), tf.float64)
        self.I = tf.Variable(self.initi, dtype=tf.float64)
        self.S = tf.Variable(self.inits, dtype=tf.float64)

        self.po_pr = tf.placeholder(tf.int64, [None, 2])
        self.po_af = tf.placeholder(tf.int32, [po_batch_size])
        self.po_num = tf.placeholder(tf.float64, [po_batch_size])
        self.po_pr_v = tf.placeholder(tf.float64, [None])
        self.po_pr_ma = tf.SparseTensor(indices=self.po_pr, values=self.po_pr_v, dense_shape=[po_batch_size, n])

        self.ne_pr = tf.placeholder(tf.int64, [None, 2])
        self.ne_af = tf.placeholder(tf.int32, [ne_batch_size])
        self.ne_num = tf.placeholder(tf.float64, [ne_batch_size])
        self.ne_pr_v = tf.placeholder(tf.float64, [None])
        self.ne_pr_ma = tf.SparseTensor(indices=self.ne_pr, values=self.ne_pr_v, dense_shape=[ne_batch_size, n])

        self.po_pr_i = tf.sparse_tensor_dense_matmul(self.po_pr_ma, self.I)
        self.po_af_s = []
        for i in range(po_batch_size):
            self.po_af_s.append(self.S[self.po_af[i]])

        self.ne_pr_i = tf.sparse_tensor_dense_matmul(self.ne_pr_ma, self.I)
        self.ne_af_s = []
        for i in range(ne_batch_size):
            self.ne_af_s.append(self.S[self.ne_af[i]])

        self.loss_po = tf.zeros([], tf.float64)
        self.loss_ne = tf.zeros([], tf.float64)

        self.one = tf.ones([], tf.float64)
        for i in range(po_batch_size):
            self.loss_po = self.loss_po - self.po_num[i] * \
                           tf.log(self.one - tf.exp(-self.lmd * tf.reduce_sum(self.po_pr_i[i] * self.po_af_s[i])) + 1e-6)

        for i in range(ne_batch_size):
            self.loss_ne = self.loss_ne + (self.ne_num[i] * self.lmd * tf.reduce_sum(self.ne_pr_i[i] * self.ne_af_s[i]))

        self.loss = self.loss_po + self.loss_ne

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.zeros = tf.zeros([n, f_len], tf.float64)
        self.filter_i = tf.assign(self.I, tf.maximum(self.I, self.zeros))
        self.filter_s = tf.assign(self.S, tf.maximum(self.S, self.zeros))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def opt(self):

        '''
        self.sess.run(tf.assign(self.I, init_i))
        self.sess.run(tf.assign(self.S, init_s))
        '''

        pre_loss = 0.0
        time_point = time.time()
        cou = 0
        while cou < ite_num:
            po_pr, po_pr_v, po_af, po_num, ne_pr, ne_pr_v, ne_af, ne_num = get_batch()
            self.sess.run(self.train_op, feed_dict={self.po_pr: po_pr, self.po_af: po_af, self.po_num: po_num,
                                                    self.ne_pr: ne_pr, self.ne_af: ne_af, self.ne_num: ne_num,
                                                    self.po_pr_v: po_pr_v, self.ne_pr_v: ne_pr_v})
            self.sess.run(self.filter_i)
            self.sess.run(self.filter_s)
            cou = cou + 1
            if cou % 10 == 0:
                now_t = time.time()
                cur_loss = self.sess.run(self.loss, feed_dict={self.po_pr: po_pr, self.po_af: po_af, self.po_num: po_num,
                                                               self.ne_pr: ne_pr, self.ne_af: ne_af, self.ne_num: ne_num,
                                                               self.po_pr_v: po_pr_v, self.ne_pr_v: ne_pr_v})
                time_per_ite = (now_t - time_point) / 10
                loss_gain = pre_loss - cur_loss
                time_point = now_t
                print('time_per_ite:', time_per_ite, 'ite_count:', cou, 'pre_loss:', pre_loss, 'cur_loss', cur_loss, 'loss_gain:', loss_gain)
                pre_loss = cur_loss
            if cou % 5000 == 0:
                now_i = self.sess.run(self.I)
                now_s = self.sess.run(self.S)
                output_matrix('matrix_i_' + str(cou), now_i, n, f_len)
                output_matrix('matrix_s_' + str(cou), now_s, n, f_len)

        self.sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-feature', type=int, default=None)
    parser.add_argument('-l', type=int, default=None)
    parser.add_argument('-ite', type=int, default=None)
    parser.add_argument('-batch', type=int, default=None)
    parser.add_argument('-lmd', type=float, default=None)
    parser.add_argument('-lr', type=float, default=None)
    parser.add_argument('-dat', type=str, default=None)
    args = parser.parse_args()

    n = args.N
    f_len = args.feature
    l = args.l
    ite_num = args.ite
    po_batch_size = args.batch
    ne_batch_size = args.batch
    lmd = args.lmd
    lr = args.lr
    dat_file = args.dat

    p_z = [[] for _ in range(n)]
    p_o = [[] for _ in range(n)]
    load_none_dup_cascade(dat_file, p_z, p_o)

    '''
    init_i = load_matrix('results\mb_64_1e-3_l5\matrix_i_40000', n, f_len)
    init_s = load_matrix('results\mb_64_1e-3_l5\matrix_s_40000', n, f_len)
    '''

    model = LIS(n, f_len, lmd, lr)
    model.opt()
