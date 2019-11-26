import numpy as np
import random
import math

from matplotlib import pylab

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def generate_w(m, n):
    w = [0.0] * m
    for i in range(m):
        w[i] = [0.0] * n
        for j in range(n):
            w[i][j] = rand(-1,1)
            # w[i][j] = rand(-0.69, 1)  # rand(-1,1) #
    return w


def generate_b(m):
    b = [0.0] * m
    for i in range(m):
        b[i] = rand(-1,1)
        # b[i] = rand(-2.409, 0.02)  # rand(-1,1)#
    return b


def fit_function(x, deriv=False):
    if deriv == True:
        # return x*(1-x)
        return 1 - np.tanh(x) * np.tanh(x)  # tanh函数的导数
    return np.tanh(x)
    # return 1/(1+np.exp(-x))


class BPNetwork:
    def __init__(self):
        self.input_n = 0
        self.input_cells = []
        self.output_n = 0
        self.output_cells = []
        self.input_w = []
        self.output_w = []
        self.hidden_ns = []
        self.hidden_ws = []
        self.hidden_bs = []
        self.output_b = []
        self.hidden_results = []
        self.output_deltas = []
        self.hidden_deltases = []

    def setup(self, input_n, output_n, hidden_set):
        # 初始化各种参数
        # self.input_n = input_n + 1
        self.input_n = input_n
        self.output_n = output_n
        self.hidden_ns = [0.0] * len(hidden_set)
        # for i in range(len(hidden_set)):
        #     self.hidden_ns[i] = hidden_set[i] + 1
        for i in range(len(hidden_set)):
            self.hidden_ns[i] = hidden_set[i]
        self.input_cells = [1.0] * self.input_n
        self.output_cells = [1.0] * self.output_n
        # 初始化weights和bias
        self.input_w = generate_w(self.input_n, self.hidden_ns[0])
        self.hidden_ws = [0.0] * (len(self.hidden_ns) - 1)
        for i in range(len(self.hidden_ns) - 1):
            self.hidden_ws[i] = generate_w(self.hidden_ns[i], self.hidden_ns[i + 1])
        self.output_w = generate_w(self.hidden_ns[len(self.hidden_ns) - 1], self.output_n)
        self.output_b = generate_b(self.output_n)
        self.hidden_bs = [0.0] * len(self.hidden_ns)
        for i in range(len(self.hidden_ns)):
            self.hidden_bs[i] = generate_b(self.hidden_ns[i])

        self.hidden_results = [0.0] * (len(self.hidden_ns))

    def forward_propagate(self, input):
        for i in range(len(input)):
            self.input_cells[i] = input[i]
        # 输入层
        self.hidden_results[0] = [0.0] * self.hidden_ns[0]
        for h in range(self.hidden_ns[0]):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_w[i][h] * self.input_cells[i]
            self.hidden_results[0][h] = fit_function(total + self.hidden_bs[0][h])
        # 隐藏层
        for k in range(len(self.hidden_ns) - 1):
            self.hidden_results[k + 1] = [0.0] * self.hidden_ns[k + 1]
            for h in range(self.hidden_ns[k + 1]):
                total = 0.0
                for i in range(self.hidden_ns[k]):
                    total += self.hidden_ws[k][i][h] * self.hidden_results[k][i]
                self.hidden_results[k + 1][h] = fit_function(total + self.hidden_bs[k + 1][h])
        # 输出层
        for h in range(self.output_n):
            total = 0.0
            for i in range(self.hidden_ns[len(self.hidden_ns) - 1]):
                total += self.output_w[i][h] * self.hidden_results[len(self.hidden_ns) - 1][i]
            self.output_cells[h] = fit_function(total + self.output_b[h])

        return self.output_cells[:]

    def get_deltas(self, label):
        self.output_deltas = [0.0] * self.output_n
        # 输出层 deltas
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            self.output_deltas[o] = fit_function(self.output_cells[o], True) * error
        # 隐藏层deltas
        tmp_deltas = self.output_deltas
        tmp_w = self.output_w
        self.hidden_deltases = [0.0] * (len(self.hidden_ns))
        k = len(self.hidden_ns) - 1
        while k >= 0:
            self.hidden_deltases[k] = [0.0] * (self.hidden_ns[k])
            for o in range(self.hidden_ns[k]):
                error = 0.0
                for i in range(len(tmp_deltas)):
                    error += tmp_deltas[i] * tmp_w[o][i]
                self.hidden_deltases[k][o] = fit_function(self.hidden_results[k][o], True) * error
            k = k - 1
            if k >= 0:
                tmp_w = self.hidden_ws[k]
                tmp_deltas = self.hidden_deltases[k + 1]
            else:
                break

    def renew_w(self, learn):
        # 更新隐藏层→输出层的权重
        k = len(self.hidden_ns) - 1
        for i in range(self.hidden_ns[k]):
            for o in range(self.output_n):
                change = self.output_deltas[o] * self.hidden_results[k][i]
                self.output_w[i][o] += change * learn

        # 更新隐藏层的权重
        while k > 0:
            for i in range(self.hidden_ns[k - 1]):
                for o in range(self.hidden_ns[k]):
                    change = self.hidden_deltases[k][o] * self.hidden_results[k - 1][i]
                    self.hidden_ws[k - 1][i][o] += change * learn
            k = k - 1

        # 更新输入层→隐藏层权重
        for i in range(self.input_n):
            for o in range(self.hidden_ns[0]):
                change = self.hidden_deltases[0][o] * self.input_cells[i]
                self.input_w[i][o] += change * learn

    def renew_b(self, learn):
        # 更新隐藏层bias
        k = len(self.hidden_bs) - 1
        while k >= 0:
            for i in range(self.hidden_ns[k]):
                self.hidden_bs[k][i] = self.hidden_bs[k][i] + learn * self.hidden_deltases[k][i]
            k = k - 1
        # 更新输出层bias
        for o in range(self.output_n):
            self.output_b[o] += self.output_deltas[o] * learn

    def back_propagate(self, input, label, learn):
        self.forward_propagate(input)
        self.get_deltas(label)
        self.renew_w(learn)
        self.renew_b(learn)
        return self.get_loss(label, self.output_cells)

    def train(self, input_datas, labels, learn=0.05, limit=100000):
        for j in range(limit):
            error = 0.0
            for i in range(len(input_datas)):
                input = input_datas[i]
                label = labels[i]
                error += self.back_propagate(input, label, learn)

    # 计算损失值
    def get_loss(self, label, output_cell):
        error = 0.0
        for o in range(len(output_cell)):
            error += 0.5 * (label[o] - output_cell[o]) ** 2
        return error

    # 计算测试集的平均损失值
    def get_average_loss(self, datas, labels):
        error = 0
        predicate_res = []
        for i in range(len(datas)):
            predicate_res.append(self.forward_propagate(datas[i]))
            error += self.get_loss(labels[i], self.output_cells)
        error = error / len(datas)
        return error,predicate_res

    # 得到训练集
    def get_train(self):
        input_datas = []
        labels = []
        for i in range(-11, 11, 1):
            input_datas.append([i * math.pi / 10])
            labels = np.sin(input_datas)
        return input_datas, labels

    # 得到测试集
    def get_test(self):
        test = []
        test_labels = []
        error = 0
        for i in range(-100, 101, 1):
            test.append([i * math.pi / 100])
            test_labels.append(np.sin([i * math.pi / 100]))
        return test, test_labels

    # 测试
    def test(self):
        learn = 0.05
        times = 3000
        input_datas, labels = self.get_train()
        self.setup(1, 1, [9, 5])
        self.train(input_datas, labels, learn, times)

        test, test_label = self.get_test()
        error,predicate_res = self.get_average_loss(test, test_label)
        print(error)

        # 画图
        pylab.plt.scatter(input_datas, labels, marker='x', color='g', label='train set')

        x = np.arange(-1 * np.pi, np.pi, 0.01)
        x = x.reshape((len(x), 1))
        y = np.sin(x)
        pylab.plot(x, y, label='standard sinx')

        pylab.plot(test, predicate_res, label='predicate sinx, learn = '+str(learn), linestyle='--', color='r')

        pylab.plt.legend(loc='best')
        pylab.plt.show()

        '''ylabels = []
        error = 0
        for i in range(len(test)):
            ylabels.append(self.forward_propagate(test[i]))
            error += self.get_loss (test_label[i], self.output_cells)

        print("---测试误差---")
        error = error/200
        print(error)
         x = np.arange (-1.0, 1.0, 0.01)
        fig = plt.figure ()
        ax1=fig.add_subplot(111)
        ax1.set_title("tanh test")
        l1, = plt.plot (x * math.pi, np.sin (x * math.pi), color='red')
        l2, = plt.plot (test, ylabels, color='green')
        plt.legend (handles=[l1, l2, ], labels=['original', 'test predict'], loc='best')
        plt.xticks ([(-1) * np.pi , (-1) * np.pi/2 ,0, np.pi / 2, np.pi],
                    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
        plt.show ()

        zables = []
        for a in input_datas:
            zables.append (self.forward_propagate (a))
        plt.figure ()
        l3, = plt.plot (x * math.pi, np.sin (x * math.pi), color='red')
        l4, = plt.plot (input_datas, zables, color='green')
        plt.legend (handles=[l3, l4, ], labels=['original', 'train predict'], loc='best')
        plt.xticks ([(-1) * np.pi , (-1) * np.pi/2 ,0, np.pi / 2, np.pi ],
                    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
        plt.show ()'''


if __name__ == '__main__':
    nn = BPNetwork()
    nn.test()
