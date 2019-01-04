import cPickle as pickle
import tensorflow as tf
import numpy as np
import sys
import argparse
from model_atten_rnn_sever import SDPP


parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=str, default=None)
parser.add_argument('-batch', type=str, default=None)
parser.add_argument('-v_size', type=str, default=None)
parser.add_argument('-T_size', type=str, default=None)
parser.add_argument('-embedding_size', type=str, default=None)
parser.add_argument('-dropout_prob', type=str, default=None)
args = parser.parse_args()


filename_train = "./atten/train.pkl"
filename_val = "./atten/val.pkl"
filename_test = "./atten/test.pkl"
import time
tf.set_random_seed(time.time())

fr = open(filename_train, "rb")
cascade_train = pickle.load(fr)
fr.close()

fr = open(filename_val, "rb")
cascade_val = pickle.load(fr)
fr.close()

fr = open(filename_test, "rb")
cascade_test = pickle.load(fr)
fr.close()


def get_batch(cascade, step, batch_size=128):
    start = step * batch_size % len(cascade[0])
    length = cascade[2]
    length_batch = []
    x_u = cascade[0]
    x_u_batch = []
    x_t = cascade[1]
    x_t_batch = []
    x_length_mat = []
        
    for i in range(batch_size):
        id = (i + start) % len(cascade[0])
        x_t_batch.append(cascade[1][id])
        x_u_batch.append(cascade[0][id])
        length_batch.append(cascade[2][id])
    
    for i in range(len(length_batch)):
        x_length_mat.append([])
        for j in range(1, len(x_u_batch[0])):
            if j < length_batch[i]:
                x_length_mat[i].append(1.0)
            else: x_length_mat[i].append(0.0)
        
    return x_u_batch, x_t_batch, x_length_mat
    
sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
training_iters = 20 * len(cascade_train[0])
print("training_iters:", training_iters)
batch_size = int(args.batch)
learning_rate = 1e-3
v_a_size = 20
T_size = 30
embedding_size = 30
dropout_prob = 1.0

display_step = min(len(cascade_train[0])//batch_size, 100)

n_steps = len(cascade_train[0][0])

np.set_printoptions(precision=2)

model = SDPP(sess, 32, batch_size, learning_rate, n_steps, v_a_size, T_size, embedding_size, dropout_prob)
sess.graph.finalize()
step = 0
best_val_loss = 10000
best_test_loss = 1000

train_loss = []
max_try = 10
patience = max_try
time_p_start = time.time()

while step * batch_size < training_iters:
    x_u_batch, x_t_batch, x_length_mat = get_batch(cascade = cascade_train, step = step, batch_size = batch_size)
    cost = model.train_batch(x_u_batch, x_t_batch, x_length_mat)
    step = step + 1

    train_loss.append(cost)
    
    if step % display_step == 0:
        print ("step:", step)
        print ("training loss:", np.mean(train_loss))
        val_loss = []
        for val_step in range(max(len(cascade_val[0])//batch_size, 1)):
            x_u_val, x_t_val, x_length_mat_val = get_batch(cascade = cascade_val, step = val_step, batch_size = batch_size)
            loss = model.get_error(x_u_val, x_t_val, x_length_mat_val)
            val_loss.append(loss)
        test_loss = []
        MRR = []
        top1 = []
        top5 = []
        RMSE = []
        for test_step in range(max(len(cascade_test[0])//batch_size, 1)):
            x_u_test, x_t_test, x_length_mat_test = get_batch(cascade = cascade_test, step = test_step, batch_size = batch_size)
            loss = model.get_error(x_u_test, x_t_test, x_length_mat_test)
            test_loss.append(loss)
            MRR_result, top1_result, top5_result, RMSE_t, bre = model.get_result(x_u_test, x_t_test, x_length_mat_test)
            if bre == 1:
                break
            MRR.append(MRR_result)
            top1.append(top1_result)
            top5.append(top5_result)
            RMSE.append(RMSE_t)
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try
            
        print ("last test error:", np.mean(test_loss))
        print ("last val error:", np.mean(val_loss))
        print ("last test MRR:", np.mean(MRR))
        print ("last test Acc@1:", np.mean(top1))
        print ("last test Acc@5:", np.mean(top5))
        print ("last test RMSE:", np.mean(RMSE))
        
        
        train_loss = []
        patience -= 1
        print(patience)
        if not patience:
            break
        if bre == 1:
            break
        
        


model.save_model("save_3rd/model.ckpt")
time_p = (time.time() - time_p_start)/step
print("Time per step: ", time_p)


test_loss = []
MRR = []
top1 = []
top5 = []
RMSE = []
cnt = []
for test_step in range(max(len(cascade_test[0])//batch_size, 1)):
    x_u_test, x_t_test, x_length_mat_test = get_batch(cascade = cascade_test, step = test_step, batch_size = batch_size)
    #test_loss.append(model.get_error(x_u_test, x_t_test, x_length_mat_test))
    loss = model.get_error(x_u_test, x_t_test, x_length_mat_test)
    test_loss.append(loss)
    MRR_result, top1_result, top5_result, RMSE_t, bre, cnt_all = model.predict(x_u_test, x_t_test, x_length_mat_test)
        
    MRR.append(MRR_result)
    top1.append(top1_result)
    top5.append(top5_result)
    RMSE.append(RMSE_t)
    cnt.append(cnt_all)
print ("last test error:", np.mean(test_loss))
print ("last test MRR:", np.sum(MRR) / np.sum(cnt))
print ("last test Acc@1:", np.sum(top1) / np.sum(cnt))
print ("last test Acc@5:", np.sum(top5) / np.sum(cnt))
print ("last test RMSE:", (np.sum(RMSE) / np.sum(cnt)) ** 0.5)


result = []
result.append(test_loss)
result.append(MRR)
result.append(top1)
result.append(RMSE)
result.append(time_p)
result.append(train_loss)
result.append(val_loss)

out_file = "./cp_exp_out_batch_size"+ str(batch_size) +".pkl" 

with open(out_file,'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)








    
    
    
    
    
    
