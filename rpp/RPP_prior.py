import tensorflow as tf
import numpy as np
import argparse
import time
import math
import random


def get_batch():
    sa_list = []
    for _ in range(batch_size):
        sa_list.append(random.choice(range(n)))
    max_len = 0
    for s in sa_list:
        max_len = max(max_len, len(time_l[s]))
    td = []
    nd = []
    bo = []
    for s in sa_list:
        nd.append(len(time_l[s]))
        td_t = []
        bo_t = []
        for t in time_l[s]:
            td_t.append(t)
            bo_t.append(1.0)
        for _ in range(max_len - len(time_l[s])):
            td_t.append(1.0)
            bo_t.append(0.0)
        td.append(td_t)
        bo.append(bo_t)
    td = np.array(td, float)
    nd = np.array(nd, float)
    bo = np.array(bo, float)
    sa = np.array(sa_list, int)
    return td, nd, bo, sa


class RPP:
    def __init__(self, N, T, m, lr):
        self.N = N
        self.T = T
        self.m = m
        self.lr = lr

        self.init_mu = tf.random_uniform([self.N], minval=6.5, maxval=8.0)
        self.init_sigma = tf.random_uniform([self.N], minval=0.8, maxval=2.2)
        self.init_alpha = tf.random_uniform([], minval=5.0, maxval=6.0)
        self.init_beta = tf.random_uniform([], minval=5.5, maxval=8.0)
        self.mu = tf.Variable(self.init_mu, dtype=tf.float32)
        self.sigma = tf.Variable(self.init_sigma, dtype=tf.float32)
        self.alpha = tf.Variable(self.init_alpha, dtype=tf.float32)
        self.beta = tf.Variable(self.init_beta, dtype=tf.float32)

        self.td = tf.placeholder(tf.float32, [batch_size, None])
        self.nd = tf.placeholder(tf.float32, [batch_size])
        self.bo = tf.placeholder(tf.float32, [batch_size, None])
        self.sa = tf.placeholder(tf.int32, [batch_size])

        self.loss = batch_size * (self.alpha * tf.log(self.beta) - tf.lgamma(self.alpha))
        self.fd = []
        self.tao = []
        self.tao_d = []
        for i in range(batch_size):
            self.loss = self.loss + tf.reduce_sum(tf.log(tf.linspace(start=self.m, stop=self.m + self.nd[i] - 1,
                                                                     num=tf.cast(self.nd[i], tf.int32))))
            self.tao_d.append((tf.log(self.td[i] + 1e-20) - self.mu[self.sa[i]]) / self.sigma[self.sa[i]])
            self.fd.append(tf.exp(-self.tao_d[i] * self.tao_d[i] / 2.0) / (tf.sqrt(2.0 * math.pi) * self.sigma[self.sa[i]] * self.td[i]))
            self.tao.append((tf.log(self.T + 1e-20) - self.mu[self.sa[i]]) / self.sigma[self.sa[i]])
        self.fd = tf.convert_to_tensor(self.fd)
        self.tao = tf.convert_to_tensor(self.tao)
        self.tao_d = tf.convert_to_tensor(self.tao_d)

        self.Fd = (1.0 + tf.erf(self.tao_d / tf.sqrt(2.0))) / 2.0
        self.FdT = (1.0 + tf.erf(self.tao / tf.sqrt(2.0))) / 2.0
        self.Xd = (self.m + self.nd) * self.FdT - tf.reduce_sum(self.Fd * self.bo, 1)
        self.loss = self.loss + tf.reduce_sum(tf.log(self.fd + 1e-20) * self.bo)
        self.loss = self.loss + tf.reduce_sum(tf.lgamma(self.alpha + self.nd))
        self.loss = self.loss - tf.reduce_sum((self.alpha + self.nd) * tf.log(self.beta + self.Xd + 1e-20))
        self.loss = -self.loss

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def opt(self):

        pre_loss = 0.0
        time_point = time.time()
        cou = 0
        while cou < ite_num:
            td, nd, bo, sa = get_batch()
            self.sess.run(self.train_op, feed_dict={self.nd: nd, self.td: td, self.bo: bo, self.sa: sa})
            cou = cou + 1
            if cou % 1000 == 0:
                now_t = time.time()
                cur_loss = self.sess.run(self.loss, feed_dict={self.nd: nd, self.td: td, self.bo: bo, self.sa: sa})
                time_per_k_ite = (now_t - time_point)
                loss_gain = pre_loss - cur_loss
                time_point = now_t
                print('time_per_k_ite:', time_per_k_ite, 'ite_count:', cou, 'pre_loss:', pre_loss, 'cur_loss', cur_loss, 'loss_gain:', loss_gain)
                pre_loss = cur_loss
            if cou % 1000 == 0:
                out = open('output_prior_' + str(cou) + '.txt', 'w')
                final_alpha = self.sess.run(self.alpha)
                final_beta = self.sess.run(self.beta)
                final_mu = self.sess.run(self.mu)
                final_sigma = self.sess.run(self.sigma)
                out.write(str(final_alpha) + '\t' + str(final_beta) + '\n')
                for i in range(n):
                    tao = (math.log(self.T) - final_mu[i]) / final_sigma[i]
                    X = (self.m + len(time_l[i])) * gs_cumulative_dt(tao)
                    for t in time_l[i]:
                        tao = (math.log(t) - final_mu[i]) / final_sigma[i]
                        X = X - gs_cumulative_dt(tao)
                    out.write(str(final_mu[i]) + '\t' + str(final_sigma[i]) + '\t' +
                              str(len(time_l[i])) + '\t' + str(X) + '\n')
                out.close()


def gs_cumulative_dt(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def predict(m, nd, T, alpha, beta, mu, sigma, X, t):
    tao_t = (math.log(t) - mu) / sigma
    tao_T = (math.log(T) - mu) / sigma
    Y = gs_cumulative_dt(tao_t) - gs_cumulative_dt(tao_T)
    return (m + nd) * math.pow((beta + X) / (beta + X -Y), alpha + nd) - m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-T', type=float, default=None)
    parser.add_argument('-m', type=float, default=None)
    parser.add_argument('-dat', type=str, default=None)
    parser.add_argument('-lr', type=float, default=None)
    parser.add_argument('-ite', type=int, default=None)
    parser.add_argument('-batch', type=int, default=None)
    args = parser.parse_args()

    n = args.N
    ite_num = args.ite
    batch_size = args.batch

    time_l = []
    file = open(args.dat, 'r')
    for _ in range(n):
        line = file.readline().split('\t')
        li_len = len(line)
        td = []
        for i in range(li_len):
            t = float(line[i])
            if t > args.T:
                break
            td.append(t)
        time_l.append(td)
    file.close()

    model = RPP(n, args.T, args.m, args.lr)
    model.opt()

