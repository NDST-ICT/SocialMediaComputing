import cPickle as pickle
import argparse

#import pickle
parser = argparse.ArgumentParser()
parser.add_argument('-dat_tra', type=str, default=None)
parser.add_argument('-dat_tes', type=str, default=None)
parser.add_argument('-dat_val', type=str, default=None)

args = parser.parse_args()
filename_train = args.dat_tra
filename_val = args.dat_val
filename_test = args.dat_tes

fr = open(filename_train, "rb")
cascade_tr = fr.readlines()
fr.close()

fr = open(filename_test, "rb")
cascade_te = fr.readlines()
fr.close()

fr = open(filename_val, "rb")
cascade_va = fr.readlines()
fr.close()

max_length = 0

for i in cascade_tr:
    cnt = (len(i.split(','))-1) // 2
    max_length = max(max_length, cnt)
    #print(cnt)
for i in cascade_te:
    cnt = (len(i.split(','))-1) // 2
    max_length = max(max_length, cnt)
    #print(cnt)
for i in cascade_va:
    cnt = (len(i.split(','))-1) // 2
    max_length = max(max_length, cnt)
    #print(cnt)

cascade_final_train = []
cascade_final_val = []
cascade_final_test = []
x_u = []
x_t = []
x_length = []

for i in range(len(cascade_tr)):
    x_u.append([])
    x_t.append([])    
    term = cascade_tr[i].split(",")
    for j in range(0, (len(term) - 1) // 2):
        x_u[i].append(int(term[2 * j + 1])) 
        x_t[i].append([float(term[2 * j + 2])])       
        

for i in range(len(x_u)):
    x_length.append(len(x_u[i]))
    while len(x_u[i]) < max_length:
        x_u[i].append(32)
        x_t[i].append([0])
        #i[0].append([32, 0])
    #print(len(x_u[i]), len(x_t[i]))


cascade_final_train.append(x_u)
cascade_final_train.append(x_t)
cascade_final_train.append(x_length)

#print (cascade)

x_u = []
x_t = []
x_length = []
for i in range(len(cascade_te)):
    x_u.append([])
    x_t.append([])    
    term = cascade_te[i].split(",")
    for j in range(0, (len(term) - 1) // 2):
        x_u[i].append(int(term[2 * j + 1])) 
        x_t[i].append([float(term[2 * j + 2])])       
        

for i in range(len(x_u)):
    x_length.append(len(x_u[i]))
    while len(x_u[i]) < max_length:
        x_u[i].append(32)
        x_t[i].append([0])
        #i[0].append([32, 0])
    #print(len(x_u[i]), len(x_t[i]))


cascade_final_test.append(x_u)
cascade_final_test.append(x_t)
cascade_final_test.append(x_length)

#print (cascade)

x_u = []
x_t = []
x_length = []
for i in range(len(cascade_va)):
    x_u.append([])
    x_t.append([])    
    term = cascade_va[i].split(",")
    for j in range(0, (len(term) - 1) // 2):
        x_u[i].append(int(term[2 * j + 1])) 
        x_t[i].append([float(term[2 * j + 2])])       
        

for i in range(len(x_u)):
    x_length.append(len(x_u[i]))
    while len(x_u[i]) < max_length:
        x_u[i].append(32)
        x_t[i].append([0])
        #i[0].append([32, 0])
    #print(len(x_u[i]), len(x_t[i]))


cascade_final_val.append(x_u)
cascade_final_val.append(x_t)
cascade_final_val.append(x_length)

#print (cascade)

out_file_train = "./atten/train.pkl"
out_file_val = "./atten/val.pkl"
out_file_test = "./atten/test.pkl"

with open(out_file_train,'wb') as f:
    pickle.dump(cascade_final_train, f, pickle.HIGHEST_PROTOCOL)

with open(out_file_val,'wb') as f:
    pickle.dump(cascade_final_val, f, pickle.HIGHEST_PROTOCOL)

with open(out_file_test,'wb') as f:
    pickle.dump(cascade_final_test, f, pickle.HIGHEST_PROTOCOL)

#print("finish")