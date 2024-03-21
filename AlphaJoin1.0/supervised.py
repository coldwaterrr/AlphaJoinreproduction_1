from datetime import datetime
import random
import os
import numpy as np
import torch
import pickle
from models import ValueNet

shortToLongPath = '../resource/shorttolong'
predicatesEncodeDictPath = './predicatesEncodedDict'


class data:
    def __init__(self, state, time):
        self.state = state
        self.time = time
        self.label = 0


class supervised:
    def __init__(self, args):
        # Read dict predicatesEncoded
        # 读取谓词预测编码
        f = open(predicatesEncodeDictPath, 'r')
        a = f.read()
        self.predicatesEncodeDict = eval(a)
        # print(self.predicatesEncodeDict)
        f.close()

        # Read all tablenames and get tablename-number mapping
        # 读取所有表名并获取表名-编号映射
        tables = []
        f = open(shortToLongPath, 'r')
        a = f.read()
        short_to_long = eval(a)
        f.close()
        for i in short_to_long.keys():
            tables.append(i)
        tables.sort()
        self.table_to_int = {}
        for i in range(len(tables)):
            self.table_to_int[tables[i]] = i

        # The dimension of the network input vector
        # 网络输入向量的维度
        self.num_inputs = len(tables) * len(tables) + len(self.predicatesEncodeDict["1a"])
        # The dimension of the vector output by the network
        # 网络输出向量的维度
        self.num_output = 5
        self.args = args
        self.right = 0

        # build up the network
        self.value_net = ValueNet(self.num_inputs, self.num_output)
        self.actor_net = ValueNet(self.num_inputs, self.num_output)
        if self.args.cuda:
            self.actor_net.cuda()
        # check some dir
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.dataList = []
        self.testList = []

    # Parsing query plan
    # 解析查询计划,计划编码的过程,最后得到论文中图一右边的图
    def hint2matrix(self, hint):  # ( ( ( ( ( mc ( ci rt ) ) cn ) t ) chn ) ct )
        tablesInQuery = hint.split(" ")
        matrix = np.mat(np.zeros((len(self.table_to_int), len(self.table_to_int))))
        stack = []
        difference = 0
        for i in tablesInQuery:
            if i == ')':
                tempb = stack.pop()
                tempa = stack.pop()
                _ = stack.pop()
                b = tempb.split('+')
                a = tempa.split('+')
                b.sort()
                a.sort()
                indexb = self.table_to_int[b[0]]
                indexa = self.table_to_int[a[0]]
                matrix[indexa, indexb] = (len(tablesInQuery) + 2) / 3 - difference
                difference += 1
                stack.append(tempa + '+' + tempb)
                # print(stack)
            else:
                stack.append(i)
        return matrix

    # Divide training set and test set
    def pretreatment(self, path):
        # Load data uniformly and randomly select for training
        # 统一加载数据并随机选择进行训练
        file_test = open(path)
        line = file_test.readline()
        while line:
            queryName = line.split(",")[0].encode('utf-8').decode('utf-8-sig').strip()  # strip默认去头尾多余空格
            hint = line.split(",")[1]
            matrix = self.hint2matrix(hint)
            predicatesEncode = self.predicatesEncodeDict[queryName]
            state = matrix.flatten().tolist()[0]
            state = state + predicatesEncode
            runtime = line.split(",")[2].strip()
            if runtime == 'timeout':
                runtime = 'timeout'  # Depends on your settings
            else:
                runtime = int(float(runtime))
            temp = data(state, runtime)
            self.dataList.append(temp)
            line = file_test.readline()

        self.dataList.sort(key=lambda x: x.time, reverse=False)
        for i in range(self.dataList.__len__()):
            self.dataList[i].label = int(i / (self.dataList.__len__() / self.num_output + 1))
            # print(self.dataList[i].label)
        for i in range(int(self.dataList.__len__() * 0.3)):
            index = random.randint(0, len(self.dataList) - 1)
            temp = self.dataList.pop(index)
            self.testList.append(temp)

        print("size of test set:", len(self.testList), "\tsize of train set:", len(self.dataList))
        testpath = "testdata.sql"
        file_test = open(testpath, 'wb')
        pickle.dump(len(self.testList), file_test)  # 通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储
        for value in self.testList:
            pickle.dump(value, file_test)
        file_test.close()

        trainpath = "traindata.sql"
        file_train = open(trainpath, 'wb')
        pickle.dump(len(self.dataList), file_train)
        for value in self.dataList:
            pickle.dump(value, file_train)
        file_train.close()

    # functions to train the network
    # 训练网络的函数
    def supervised(self):
        self.load_data()
        optim = torch.optim.SGD(self.value_net.parameters(), lr=0.01)
        # loss_func = torch.nn.MSELoss()
        # loss_func = torch.nn.CrossEntropyLoss()
        loss_func = torch.nn.NLLLoss()
        loss1000 = 0
        count = 0

        for step in range(1, 16000001):
            index = random.randint(0, len(self.dataList) - 1)
            state = self.dataList[index].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = torch.log(self.value_net(state_tensor) + 1e-10)
            predictionRuntime = predictionRuntime.view(1, -1)

            label = []
            label.append(self.dataList[index].label)
            label_tensor = torch.tensor(label)

            loss = loss_func(predictionRuntime, label_tensor)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss1000 += loss.item()
            if step % 1000 == 0:
                print('[{}]  Epoch: {}, Loss: {:.5f}'.format(datetime.now(), step, loss1000))
                loss1000 = 0
                # self.test_network() ??
                print('[{}]  Epoch: {}, Loss: {:.5f}'.format(datetime.now(), step, loss1000))
            if step % 200000 == 0:
                torch.save(self.value_net.state_dict(), self.args.save_dir + 'supervised.pt')
                # self.test_network()

    # functions to test the network
    # 测试网络的函数
    def test_network(self):
        self.load_data()
        model_path = self.args.save_dir + 'supervised.pt'
        # self.actor_net = self.value_net(self.num_inputs, self.num_output)
        self.actor_net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.actor_net.eval()

        correct = 0
        for step in range(self.testList.__len__()):
            state = self.testList[step].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = self.actor_net(state_tensor)
            prediction = predictionRuntime.detach().cpu().numpy()
            maxindex = np.argmax(prediction)
            label = self.testList[step].label
            # print(maxindex, "\t", label)
            if maxindex == label:
                correct += 1
        print(correct, self.testList.__len__(), correct / self.testList.__len__(), end=' ')

        correct1 = 0
        for step in range(self.dataList.__len__()):
            state = self.dataList[step].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = self.actor_net(state_tensor)
            # prediction = predictionRuntime.detach().cpu().numpy()[0]
            prediction = predictionRuntime.detach().cpu().numpy()
            maxindex = np.argmax(prediction)
            label = self.dataList[step].label
            # print(maxindex, "\t", label)
            if maxindex == label:
                correct1 += 1
        print(correct1, self.dataList.__len__(), correct1 / self.dataList.__len__())
        self.right = correct / self.testList.__len__()

    def load_data(self):
        if self.dataList.__len__() != 0:
            return
        testpath = "testdata.sql"
        file_test = open(testpath, 'rb')
        l = pickle.load(file_test)
        for _ in range(l):
            self.testList.append(pickle.load(file_test))
        file_test.close()

        trainpath = "traindata.sql"
        file_train = open(trainpath, 'rb')
        l = pickle.load(file_train)
        for _ in range(l):
            self.dataList.append(pickle.load(file_train))
        file_train.close()
