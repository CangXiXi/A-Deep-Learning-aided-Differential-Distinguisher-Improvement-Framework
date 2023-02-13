import random

import numpy as np

from Github import train_nets as tn

list = list(range(0,16))
acc = np.zeros(100)
M = np.zeros((16,100))
S = np.zeros(16)


for i in range(100):
    list2 = random.sample(list,4)
    list3 = np.sort(list2)
    acc[i] = tn.train_speck_distinguisher(num_epochs=10, num_rounds=6, depth=5,bit=list3,bit_num=8,wdir='./')
    print(acc)
    for j in range(16):
        if j in list2:
            M[j,i] = 1

for i in range(16):
    for j in range(100):
        S[i] = S[i]+(M[j,i]*acc[100])

np.savez('data',acc=acc,M=M,S=S)