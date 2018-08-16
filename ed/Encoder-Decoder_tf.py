import numpy as np
import scipy.sparse
import tensorflow as tf
import argparse


def output_matrix(path, ma, n, m):
    out_ma_f = open(path, 'w')
    for i in range(n):
        for j in range(m):
            out_ma_f.write(str(ma[i][j]) + '\t')
        out_ma_f.write('\n')
    out_ma_f.close()


class ED:
    def __init__(self, n, k, A):
        self.initw = tf.cast(tf.random_uniform([n, k], minval=0.0, maxval=1.0), tf.float64)
        self.initz = tf.cast(tf.random_uniform([k, n], minval=0.0, maxval=1.0), tf.float64)
        self.W = tf.Variable(self.initw, dtype=tf.float64)
        self.Z = tf.Variable(self.initz, dtype=tf.float64)
        indices = []
        for i in range(A.nnz):
            indices.append([A.row[i], A.col[i]])
        self.A = tf.SparseTensor(indices=indices, values=A.data, dense_shape=[n, n])

        self.up_w = tf.sparse_tensor_dense_matmul(self.A, tf.transpose(self.Z))
        self.down_w = tf.matmul(self.W, tf.matmul(self.Z, self.Z, transpose_b=True)) + tf.sparse_tensor_dense_matmul(self.A, tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.A), self.W))
        self.down_w_true = tf.maximum(self.down_w, 1e-80)
        self.up_z = tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.A), self.W))
        self.down_z = tf.matmul(tf.matmul(self.W, self.W, transpose_a=True), self.Z) + self.Z
        self.down_z_true = tf.maximum(self.down_z, 1e-80)
        self.new_w = 2.0*self.W*self.up_w / self.down_w_true
        self.new_z = 2.0*self.Z*self.up_z / self.down_z_true

        self.epsilon = tf.placeholder(tf.float64)
        self.New_w = tf.placeholder(tf.float64, shape=self.W.shape)
        self.New_z = tf.placeholder(tf.float64, shape=self.Z.shape)
        self.update_w = tf.assign(self.W, self.epsilon*self.W + (1-self.epsilon)*self.New_w)
        self.update_z = tf.assign(self.Z, self.epsilon*self.Z + (1-self.epsilon)*self.New_z)
        self.filter_w = tf.assign(self.W, tf.where(self.W > 1e-80, self.W, tf.zeros(self.W.shape, self.W.dtype)))
        self.filter_z = tf.assign(self.Z, tf.where(self.Z > 1e-80, self.Z, tf.zeros(self.Z.shape, self.Z.dtype)))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.pre = np.array(0 for _ in range(n))

    def opt(self):
        cou = 0
        epsilon = 0.0
        while cou < 500:
            new_w = self.sess.run(self.new_w)
            new_z = self.sess.run(self.new_z)
            self.sess.run(self.update_w, feed_dict={self.New_w: new_w, self.epsilon: epsilon})
            self.sess.run(self.update_z, feed_dict={self.New_z: new_z, self.epsilon: epsilon})
            self.sess.run(self.filter_w)
            self.sess.run(self.filter_z)
            cou = cou + 1
            if cou % 100 == 1:
                epsilon = epsilon + 0.1999
            print("ite_count:", cou)
        final_w = self.sess.run(self.W)
        final_z = self.sess.run(self.Z)

        output_matrix('matrix_w', final_w, n, k)
        output_matrix('matrix_z', final_z, k, n)

        self.pre = np.argmax(final_w, 1)
        self.sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=None)
    args = parser.parse_args()

    A = scipy.sparse.load_npz('net.npz')
    A = A.tocoo()
    n = A.shape[0]
    k = args.k

    model = ED(n, k, A)
    model.opt()
    np.save('results.npy', model.pre)

    label = list(model.pre)
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

