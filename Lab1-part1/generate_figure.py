import matplotlib.pyplot as plt
import numpy as np
import random
import math
import lab1_BPNetwork_1 as bpNetwork


def learn_loss():

    x = []
    y=[0.0]*4
    for i in range(31):
        x.append(i*100)
    learn = [0.01,0.06,0.1,0.6]
    for j in range(4):
        bpNet = bpNetwork.BPNetwork ()
        train_data, train_label = bpNet.get_train ()
        test_data, test_label = bpNet.get_test ()
        bpNet.setup(1,1,[6])
        y[j] = [0.0]*31
        y[j][0] = bpNet.get_average_loss (test_data, test_label)
        for i in range (30):
            bpNet.train(train_data,train_label,learn[j],100)
            y[j][i+1] = bpNet.get_average_loss(test_data,test_label)
    for i in range(4):
        print("当learning rate = "+str(learn[i])+" 时，平均loss为："+ str(y[i][30]))
    labels = ['rate=0.03', 'rate=0.06', 'rate=0.1', 'rate=0.6',]
    title = "loss with different learning rates"
    draw(x,y,labels,title)

def layer_size_different():

    x = []
    y = [0.0] * 4
    for i in range (31):
        x.append (i * 100)

    for i in range(4):
        tmp = [0.0] * (i+1)
        for j in range(i+1):
            tmp[j] = 10
        bpNet = bpNetwork.BPNetwork ()
        train_data, train_label = bpNet.get_train ()
        test_data, test_label = bpNet.get_test ()
        bpNet.setup(1,1,tmp)
        y[i]=[0.0]*31
        y[i][0] = bpNet.get_average_loss(test_data, test_label)
        for k in range(30):
            bpNet.train (train_data, train_label, 0.05, 100)
            y[i][k+1] = bpNet.get_average_loss(test_data, test_label)

        print ("当隐藏层设置为： " + str (tmp) + " 时，平均loss为：" + str(y[i][30]))

    title = "loss with different hidden layers"
    labels = ['layer_size=1', 'layer_size=2', 'layer_size=3', 'layer_size=4',]
    draw(x,y,labels,title)

def hidden_size_dif():

    x = []
    y = [0.0] * 4
    for i in range (31):
        x.append (i * 100)
    size = [6,20,50,200]
    for i in range (4):
        bpNet = bpNetwork.BPNetwork ()
        train_data, train_label = bpNet.get_train ()
        test_data, test_label = bpNet.get_test ()
        tmp = [ ]
        tmp.append(size[i])
        bpNet.setup (1, 1, tmp)
        y[i] = [0.0] * 31
        y[i][0] = bpNet.get_average_loss (test_data, test_label)
        for k in range (30):
            bpNet.train (train_data, train_label, 0.05, 100)
            y[i][k + 1] = bpNet.get_average_loss (test_data, test_label)
        print ("当只有一层隐藏层，该隐藏层大小为： " + str(tmp) + " 时，平均loss为：" + str(y[i][30]))
    title = "loss with different hidden size"
    labels = ['hidden_size=6', 'hidden_size=20', 'hidden_size=50', 'hidden_size=200', ]
    draw (x, y, labels, title)


def draw(x,y,labels,title):
    fig = plt.figure ()
    ax1 = fig.add_subplot (111)
    ax1.set_title (title)
    plt.xlabel ("iteration times")
    plt.ylabel ("loss")
    l1, = plt.plot (x, y[0], color='red',label=labels[0])
    l2, = plt.plot (x, y[1], color='green',label=labels[1])
    l3, = plt.plot (x, y[2], color='blue',label=labels[2])
    l4 = plt.plot (x, y[3], color='yellow',label=labels[3])
    plt.legend (loc='upper center', bbox_to_anchor=(0.6, 0.95), ncol=2, fancybox=True, shadow=True)
    #plt.legend (handles=[l1, l2, l3, l4, ], labels=labels ,loc='best',ncol=2)
    plt.xticks ([0, 500, 1000, 1500, 2000, 2500, 3000],
                    [r'$0$', r'$500$', r'$1000$', r'$1500$', r'$2000$', r'$2500$', r'$3000$'])
    plt.show ()

if __name__ == '__main__':
    layer_size_different()
    learn_loss()
    hidden_size_dif()



