import tensorflow as tf
import numpy as np
import argparse
import time
import math


class RPP:
    def __init__(self, T, m, lr):
        self.T = T
        self.m = m
        self.lr = lr

        self.init_mu = tf.random_uniform([], minval=6.5, maxval=8.0)
        self.init_sigma = tf.random_uniform([], minval=0.8, maxval=2.2)
        self.mu = tf.Variable(self.init_mu, dtype=tf.float32)
        self.sigma = tf.Variable(self.init_sigma, dtype=tf.float32)

        self.td = tf.placeholder(tf.float32, [None])
        self.nd = tf.placeholder(tf.float32, [])

        self.id = tf.linspace(start=self.m, stop=self.m + self.nd - 1, num=tf.cast(self.nd, tf.int32))
        self.loss = tf.reduce_sum(tf.log(self.id))
        self.tao_d = (tf.log(self.td + 1e-20) - self.mu) / self.sigma
        self.Fd = (1.0 + tf.erf(self.tao_d / tf.sqrt(2.0))) / 2.0
        self.tao = (tf.log(self.T + 1e-20) - self.mu) / self.sigma
        self.FdT = (1.0 + tf.erf(self.tao / tf.sqrt(2.0))) / 2.0
        self.lmd = self.nd / ((self.m + self.nd) * self.FdT - tf.reduce_sum(self.Fd))
        self.fd = tf.exp(-self.tao_d * self.tao_d / 2.0) / (tf.sqrt(2.0 * math.pi) * self.sigma * self.td)
        self.loss = self.loss + tf.reduce_sum(tf.log(self.fd + 1e-20))
        self.loss = self.loss + self.nd * tf.log(self.lmd + 1e-20) - self.nd
        self.loss = -self.loss

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session()

    def opt(self, index):
        nd = len(time_l[index])
        td = np.array(time_l[index])

        pre_loss = 0.0
        time_point = time.time()
        cou = 0
        self.sess.run(tf.global_variables_initializer())

        while 1:
            self.sess.run(self.train_op, feed_dict={self.nd: nd, self.td: td})
            cou = cou + 1
            cur_loss = self.sess.run(self.loss, feed_dict={self.nd: nd, self.td: td})
            loss_gain = pre_loss - cur_loss
            if math.fabs(loss_gain) < 1e-3:
                break
            if cou % 1000 == 0:
                now_t = time.time()
                time_per_k_ite = (now_t - time_point)
                time_point = now_t
                print('time_per_k_ite:', time_per_k_ite, 'ite_count:', cou, 'pre_loss:', pre_loss, 'cur_loss', cur_loss, 'loss_gain:', loss_gain)
            pre_loss = cur_loss

        print('complete item ' + str(index))
        final_lmd = self.sess.run(self.lmd, feed_dict={self.nd: nd, self.td: td})
        final_mu = self.sess.run(self.mu)
        final_sigma = self.sess.run(self.sigma)
        out.write(str(final_lmd) + '\t' + str(final_mu) + '\t' + str(final_sigma) + '\t' + str(nd) + '\n')


def gs_cumulative_dt(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def predict(m, nd, T, lmd, mu, sigma, t):
    tao_t = (math.log(t) - mu) / sigma
    tao_T = (math.log(T) - mu) / sigma
    expo = lmd * (gs_cumulative_dt(tao_t) - gs_cumulative_dt(tao_T))
    return (m + nd) * math.exp(expo) - m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-T', type=float, default=None)
    parser.add_argument('-m', type=float, default=None)
    parser.add_argument('-dat', type=str, default=None)
    parser.add_argument('-lr', type=float, default=None)
    #parser.add_argument('-ite', type=int, default=None)
    args = parser.parse_args()

    n = args.N
    #ite_num = args.ite

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

    out = open('output.txt', 'w')
    model = RPP(args.T, args.m, args.lr)
    for i in range(n):
        model.opt(i)
    model.sess.close()
    out.close()
