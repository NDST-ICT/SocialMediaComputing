import numpy as np
import tensorflow as tf
import math
import random
from tensorflow.contrib import rnn
from sympy import  integrate,exp

class SDPP(object):
    def __init__(self, sess, n_nodes, batch_size, learning_rate, n_steps, v_a_size, T_size, embedding_size, dropout_prob):
        self.initializer = tf.random_normal_initializer(stddev=0.01)
        self.initializer1 = tf.orthogonal_initializer()
        self.initializer2 = tf.random_uniform_initializer(minval = 0,maxval = 1,dtype = tf.float32)
        self.n_nodes = n_nodes + 1
        self.n_steps = n_steps
        self.embedding_size = embedding_size
        self.n_hidden_gru = 32
        self.v_a_size = v_a_size 
        self.T_size = T_size   
        self.name = 'attention_based_rnn'
        self.dropout_prob = dropout_prob
        self.t_size = 1
        self.learning_rate = learning_rate
        self.sess = sess
        self.batch_size = batch_size
        self.cost = 0.0
        
        self.build_input()
        self.build_var()
        self.build_model()
        print("finished!")
        
        self.cost = -self.cost
        
		
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        self.saver = tf.train.Saver(max_to_keep=4)
                
        

    def build_input(self):
        self.x_u = tf.placeholder(tf.int32, shape = [self.batch_size, self.n_steps], name = "x_u")
        #(total_number of sequence, n_steps, 1)  u 
        self.x_t = tf.placeholder(tf.float32, shape = [self.batch_size, self.n_steps, self.t_size], name = "x_t")
        #(total_number of the users in a path, n_steps, t_size)  t
        self.x_length_mat = tf.placeholder(tf.float32, shape = [self.batch_size, self.n_steps - 1], name = "x_length_mat")
        #(total_number of the users in a path, n_steps)  

    def build_var(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable('embedding', initializer = self.initializer2([self.n_nodes, self.embedding_size]), dtype = tf.float32)
            with tf.variable_scope('GRU'):
                self.gru_cell = rnn.GRUCell(2*self.n_hidden_gru)
            with tf.variable_scope('attention'):
                self.w_stt = tf.get_variable('Wtt_lamna', initializer = tf.random_uniform_initializer(minval = 0,maxval = 0.1,dtype = tf.float32)([1]), dtype = tf.float32)
                self.w_t = tf.get_variable('WT_lamna', initializer = self.initializer([1, self.T_size]), dtype = tf.float32)
                self.z_t = tf.get_variable('Zt_lamna', initializer = self.initializer([1, self.embedding_size + self.t_size]), dtype = tf.float32)
                self.u_t = tf.get_variable('Ut_lamna', initializer = self.initializer([1, 2*self.n_hidden_gru]), dtype = tf.float32)
                self.v_a = tf.get_variable('v_alf', initializer = self.initializer([1, self.v_a_size]), dtype = tf.float32)
                self.w_a = tf.get_variable('w_alf', initializer = self.initializer1([self.v_a_size, self.T_size]), dtype = tf.float32)
                self.u_a = tf.get_variable('u_alf', initializer = self.initializer1([self.v_a_size, 2*self.n_hidden_gru]), dtype = tf.float32)
                self.w_TT = tf.get_variable('w_TT', initializer = self.initializer1([self.T_size, self.T_size]), dtype = tf.float32)
                self.w_Tx = tf.get_variable('w_Tx', initializer = self.initializer1([self.T_size, self.embedding_size + self.t_size]), dtype = tf.float32)
                self.w_Ts = tf.get_variable('w_Ts', initializer = self.initializer1([self.T_size, 2*self.n_hidden_gru]), dtype = tf.float32)
                self.w_gs = tf.get_variable('w_Gs', initializer = self.initializer1([self.n_nodes, 2*self.n_hidden_gru]), dtype = tf.float32)
                self.w_gT = tf.get_variable('w_GT', initializer = self.initializer1([self.n_nodes, self.T_size]), dtype = tf.float32)
                self.w_gx = tf.get_variable('w_Gx', initializer = self.initializer1([self.n_nodes, self.embedding_size + self.t_size]), dtype = tf.float32)
                self.b_s = tf.get_variable('b_es', initializer = self.initializer([1, self.v_a_size]), dtype = tf.float32)
                self.b_T = tf.get_variable('b_T', initializer = self.initializer([1, self.T_size]), dtype = tf.float32)
                self.b_G = tf.get_variable('b_G', initializer = self.initializer([1, self.n_nodes]), dtype = tf.float32)
			

    def build_model(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('attention_based_rnn') as scope:
                with tf.variable_scope('embedding'):
                    x_vector = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, self.x_u),
                                            self.dropout_prob)
                    self.w_stt = tf.abs(self.w_stt)
                    self.w_stt = tf.maximum(self.w_stt,1e-8)
                    t_sub = []
                    for i in range(self.x_t.shape[0].value):
                        t_sub.append([[0]])  #take up the position / futile
                        for j in range(1, self.x_t.shape[1].value):
                            t_sub[i].append([self.x_t[i][j][0] - self.x_t[i][j-1][0]])
                        
                    self.t_sub = t_sub         #tk-tk-1 (batch_size * n_steps * t_size)                    for i in range(self.x_t.shape[0].value):
                    
                    
                    t_sub_e = []
                    for i in range(self.x_t.shape[0].value):
                        t_sub_e.append([[0]])  #take up the position / futile
                        for j in range(1, self.x_t.shape[1].value):
                            t_sub_e[i].append([tf.exp(self.x_t[i][j][0] * self.w_stt) - tf.exp(self.x_t[i][j-1][0] * self.w_stt)])
                        
                    self.t_sub_e = t_sub_e         #tk-tk-1 (batch_size * n_steps * t_size)
                    
                    x_vector = tf.concat([x_vector, tf.log(tf.maximum(t_sub,1e-8))], 2)  ##log(tk-tk-1)!! need change
                    #(batch_size, n_steps, embedding_size + t_size)
                    x_vector_form = x_vector
                with tf.variable_scope('RNN'):
                    x_vector = tf.transpose(x_vector, [1,0,2])
                    #(n_steps, batch_size, embedding_size + t_size)
                    x_vector = tf.reshape(x_vector, [-1, self.embedding_size+self.t_size])
                    #(n_steps * batch_size, embedding_size + t_size)
                    x_vector = tf.split(x_vector, self.n_steps, 0)
                    # Split to get a list of 'n_steps' tensors of shape
                    # (batch_size, embedding_size + t_size)
                    outputs,_ = rnn.static_rnn(self.gru_cell, x_vector, dtype=tf.float32)
                    
                    hidden_states = tf.transpose(tf.stack(outputs), [1, 0, 2])
                    #(batch_size, n_steps, 2*n_hidden_gru)
                    
                    hidden_states = tf.reshape(hidden_states, [-1, self.n_steps, 2*self.n_hidden_gru])
                    #(batch_size, n_steps, 2*n_hidden_gru) ??
                    
                with tf.variable_scope('attention'):
                    
                    self.e = []
                    self.a = []
                    self.a_g = []
                    self.coh = []
                    self.s = []
                    self.T = []
                    self.f = []
                    g = []
                    
                    
                    abc_cnt=0
                    print(hidden_states.shape)
                    
                    for i in range(hidden_states.shape[0].value):
                            u_a_m_h = tf.matmul(hidden_states[i],tf.transpose(self.u_a,[1,0]))
                            self.e.append([])
                            self.a.append([])
                            self.a_g.append([])
                            self.coh.append([])
                            self.s.append([])
                            self.T.append([])
                            self.f.append([])
                            g.append([])
                            
                            self.e[i].append(tf.exp(tf.matmul(tf.tanh(tf.expand_dims(u_a_m_h[0], 0) + self.b_s), tf.transpose(self.v_a, [1,0]))))
                            self.a[i].append([1])
                            self.s[i].append(tf.expand_dims(hidden_states[i][0], 0))
                            self.T[i].append(tf.sigmoid(tf.matmul(tf.expand_dims(x_vector_form[i][0], 0), tf.transpose(self.w_Tx, [1,0])) 
                                      + tf.matmul(self.s[i][0], tf.transpose(self.w_Ts, [1,0])) + self.b_T ))
                            
                            
                            for j in range(1, hidden_states.shape[1].value):
                                coh = (tf.matmul(self.T[i][j-1], tf.transpose(self.w_t, [1,0]))[0][0]
                                            + tf.matmul(tf.expand_dims(x_vector_form[i][j-1], 0), tf.transpose(self.z_t, [1,0]))[0][0] 
                                            + tf.matmul(tf.expand_dims(hidden_states[i][j-1], 0), tf.transpose(self.u_t, [1,0]))[0][0]) 
                                self.coh[i].append(coh)
                                
                                g[i].append(tf.exp(tf.matmul(tf.expand_dims(x_vector_form[i][j-1], 0), tf.transpose(self.w_gx, [1,0]))
                                          + tf.matmul(self.T[i][j-1], tf.transpose(self.w_gT, [1,0]))
                                          + tf.matmul(self.s[i][j-1], tf.transpose(self.w_gs, [1,0])) + self.b_G))
                                sum_g = tf.reduce_sum(g[i][j-1])
                                
                                a_g = g[i][j-1][0] / sum_g 
                                
                                self.a_g[i].append(a_g)
                                self.g = g
                                self.cost += (self.x_t[i][j][0] * self.w_stt + coh - self.t_sub_e[i][j][0] * tf.exp(coh) / tf.maximum(self.w_stt,1e-8) + tf.log(tf.maximum(a_g[self.x_u[i][j]],1e-8))) * self.x_length_mat[i][j-1]
                                
                                self.e[i].append(tf.exp(tf.matmul(tf.tanh(tf.expand_dims(u_a_m_h[j], 0) + tf.matmul(self.T[i][j-1], tf.transpose(self.w_a)) + self.b_s), tf.transpose(self.v_a, [1,0]))))
                                sum_e = tf.reduce_sum(self.e[i])
                                self.a[i].append(mem / sum_e for mem in self.e[i])
                                self.s[i].append(tf.matmul(tf.transpose(tf.concat([_ for _ in self.a[i][j]], 0), [1,0]), tf.slice(hidden_states[i], [0, 0], [j+1, -1]) ))
                                self.T[i].append(tf.sigmoid(tf.matmul(tf.expand_dims(x_vector_form[i][j], 0), tf.transpose(self.w_Tx, [1,0])) 
                                          + tf.matmul(self.T[i][j-1], self.w_TT) 
                                          + tf.matmul(self.s[i][j], tf.transpose(self.w_Ts, [1,0])) + self.b_T ))
                                abc_cnt += 1
                                
                                
    def train_batch(self, x_u, x_t, x_length_mat):
        _, cost= self.sess.run([self.train_op, self.cost], 
                                    feed_dict={self.x_u:x_u, self.x_t:x_t, self.x_length_mat:x_length_mat} )

        return cost


    def get_result(self, x_u, x_t, x_length_mat):
        g, coh, w = self.sess.run([self.a_g, self.coh, self.w_stt], 
                                 feed_dict={self.x_u:x_u, self.x_t:x_t, self.x_length_mat:x_length_mat} )
        
        cnt_top1 = 0.0
        cnt_top5 = 0.0
        MRR_result = 0.0
        cnt_all = 0.0
        bre = 0
        RMSE_t = 0.0
        for i in range(self.batch_size):
            for j in range(x_length_mat[i].count(1)):
                cnt_all += 1.0
                if math.isnan(g[i][j][x_u[i][j+1]]):
                    bre = 1
                    MRR_result = 0.0
                    top1_result = 0.0
                    top5_result = 0.0
                    return MRR_result, top1_result, top5_result, RMSE_t, bre
                    
                if Acc_top_k(g[i][j], x_u[i][j+1], 1):
                    cnt_top1 += 1.0
                if Acc_top_k(g[i][j], x_u[i][j+1], 5):
                    cnt_top5 += 1.0
                MRR_result += MRR(g[i][j], x_u[i][j+1])
                t_pre = 0 
                for ij in range(1000):
                    F = random.uniform(0,1)
                    t_p = math.log(-w * math.exp(-coh[i][j]) * math.log(1.0 - F) + math.exp(w * x_t[i][j][0])) / w
                    t_pre += t_p
                t_pre /= 1000
                RMSE_t += (t_pre - x_t[i][j+1][0]) ** 2
            
        RMSE_t = (RMSE_t / cnt_all) ** 0.5
        MRR_result = MRR_result / cnt_all
        top1_result = cnt_top1 / cnt_all
        top5_result = cnt_top5 / cnt_all
        return MRR_result, top1_result, top5_result, RMSE_t, bre

    def predict(self, x_u, x_t, x_length_mat):
        g, coh, w = self.sess.run([self.a_g, self.coh, self.w_stt], 
                                 feed_dict={self.x_u:x_u, self.x_t:x_t, self.x_length_mat:x_length_mat} )
        cnt_top1 = 0.0
        cnt_top5 = 0.0
        MRR_result = 0.0
        cnt_all = 0.0
        bre = 0
        RMSE_t = 0.0
        for i in range(self.batch_size):
            temp = [x_u[i][0]]
            for casc in g[i]:
                temp.append(np.argmax(casc))
            
            for j in range(x_length_mat[i].count(1)):
                cnt_all += 1.0
                if math.isnan(g[i][j][x_u[i][j+1]]):
                    bre = 1
                    MRR_result = 0.0
                    top1_result = 0.0
                    top5_result = 0.0
                    return MRR_result, top1_result, top5_result, RMSE_t, bre
                    
                if Acc_top_k(g[i][j], x_u[i][j+1], 1):
                    cnt_top1 += 1.0
                if Acc_top_k(g[i][j], x_u[i][j+1], 5):
                    cnt_top5 += 1.0
                MRR_result += MRR(g[i][j], x_u[i][j+1])    
                     
                t_pre = 0 
                for ij in range(1000):
                    F = random.uniform(0,1)
                    t_p = math.log(-w * math.exp(-coh[i][j]) * math.log(1.0 - F) + math.exp(w * x_t[i][j][0])) / w
                    t_pre += t_p
                t_pre /= 1000
                if (t_pre - x_t[i][j+1][0]) ** 2 < 60:
                    RMSE_t += (t_pre - x_t[i][j+1][0]) ** 2
                if (t_pre - x_t[i][j+1][0]) ** 2 > 60:
                    print((t_pre - x_t[i][j+1][0]) ** 2, "bigger than 60!")
                    print(t_pre, x_t[i][j+1][0], t_pre - x_t[i][j+1][0], (t_pre - x_t[i][j+1][0]) ** 2)
                    print(t_pre, x_u[i][j-1], x_u[i][j])

        MRR_result = MRR_result 
        top1_result = cnt_top1 
        top5_result = cnt_top5 
        
        return MRR_result, top1_result, top5_result, RMSE_t, bre, cnt_all

    def get_error(self, x_u, x_t, x_length_mat):
        cost = self.sess.run([self.cost], 
                                 feed_dict={self.x_u:x_u, self.x_t:x_t, self.x_length_mat:x_length_mat} )
        return cost
    
    def save_model(self, address):
        self.saver.save(self.sess, address)
        
    def load_model(self, address):
        self.saver.restore(self.sess, address)
    



def Acc_top_k(a_g, u, k):
    g = sorted(list(a_g), reverse=True)
    top_k = g[:k]
    return (a_g[u] in top_k)
    

def MRR(a_g, u):
    g = sorted(list(a_g), reverse=True)
    return (1.0 / float(g.index(a_g[u]) + 1 ))
    

















