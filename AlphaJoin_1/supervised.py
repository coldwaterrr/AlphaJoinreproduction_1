from datetime import datetime
import random
import os
import numpy as np
import torch
import pickle
from models import ValueNet
from resnet import ResNet
import matplotlib.pyplot as plt


shortToLongPath = '../resource/shorttolong'
predicatesEncodeDictPath = './predicatesEncodedDict'


class data:
    def __init__(self, queryname, state, time):
        self.queryname = queryname
        self.state = state
        self.time = time
        self.label = 0
        # self.operaotor = operator


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
        for i in short_to_long.keys():  # 表对应的缩写写进tables里
            tables.append(i)
        # print(tables)
        tables.sort()  # 按字母顺序排序
        print(tables)
        self.table_to_int = {}
        for i in range(len(tables)):  # 所有表的数量->28个 里面包含了索引21+7
            self.table_to_int[tables[i]] = i
        # print(self.table_to_int)
        print(len(tables))

        # The dimension of the network input vector
        # 网络输入向量的维度
        self.num_inputs = len(tables) * len(tables) + len(self.predicatesEncodeDict["00000"])  # predicatesEncodeDict[queryname] 长度统一为 72+3
        # The dimension of the vector output by the network
        # 网络输出向量的维度
        self.num_output = 6
        self.args = args
        self.right = 0

        # build up the network
        # 建立网络
        self.value_net = ValueNet(self.num_inputs, self.num_output*4)  # 值网络  普通神经网络
        # self.value_net = ResNet(self.num_inputs, self.num_output)  # ResNet
        # self.actor_net = ValueNet(self.num_inputs, self.num_output)  # 动作网络？有啥区别
        if self.args.cuda:
            print("使用了GPU")
            self.value_net.cuda()
        # print("使用了GPU")
        # self.actor_net.cuda()

        # check some dir
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.dataList = []
        self.testList = []

    # Parsing query plan
    # 解析查询计划,计划编码的过程,最后得到论文中图一右边的图
    def hint2matrix(self, hint):  # ( ( ( ( ( mc ( ci rt ) ) cn ) t ) chn ) ct )
        tablesInQuery = hint.split(" ")
        matrix = np.mat(np.zeros((len(self.table_to_int), len(self.table_to_int))))  # numpy.mat()矩阵类型，这是生成一个28*28的全0矩阵
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
                # print('indexb:'+b[0], end=' ')
                # print(indexb)
                # print('indexa:'+a[0], end='  ')
                # print(indexa)
                # print(len(tablesInQuery))
                # print(difference)
                # print(tablesInQuery)
                matrix[indexa, indexb] = (len(tablesInQuery) + 2) / 3 - difference  # indexa和indexb是横坐标和纵坐标
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
        operator = {}
        while line:
            queryName = line.split(",")[0].encode('utf-8').decode('utf-8-sig').strip()  # strip默认去头尾多余空格
            hint = line.split(",")[1]  # ( ( ( ( ( mc ( ci rt ) ) cn ) t ) chn ) ct )
            matrix = self.hint2matrix(hint)  # 变成paper图一右边的图
            predicatesEncode = self.predicatesEncodeDict[queryName]
            # print(queryName, end="\t\t")
            # print(type(queryName), end='\t')
            # print(predicatesEncode, end="\t")
            # print(len(predicatesEncode))
            state = matrix.flatten().tolist()[0]
            state = state + predicatesEncode
            # print(state)
            runtime = line.split(",")[2].strip()

            print(line.split(",")[3].split('\n')[0])
            # print(type(line.split(",")[3]))
            if line.split(",")[3].split('\n')[0] == "0":
                op = "no_restrict"
                operator[queryName] = 0
            elif line.split(",")[3].split('\n')[0] == "1":
                op = "no_hash"
                operator[queryName] = 1
            elif line.split(",")[3].split('\n')[0] == "2":
                op = "no_merge"
                operator[queryName] = 2
            elif line.split(",")[3].split('\n')[0] == "3":
                op = "nestloop"
                operator[queryName] = 3

            if runtime == 'timeout':
                runtime = 'timeout'  # Depends on your settings
            else:
                runtime = int(float(runtime))
            # print(type(queryName))
            temp = data(queryName, state, runtime)
            self.dataList.append(temp)
            # print(type(self.dataList[0].queryname))
            line = file_test.readline()

        self.dataList.sort(key=lambda x: x.time, reverse=False)
        print(operator)
        for i in range(self.dataList.__len__()):
            # print(operator['03803'])
            self.dataList[i].label = (int(i / (self.dataList.__len__() / self.num_output + 1))+1) * 4 - int(operator[self.dataList[i].queryname])-1
            # print(self.dataList[i].queryname)
            # print(operator[])
            print(self.dataList[i].label)
            # print(self.dataList.__len__())
            # print(self.num_output)
        for i in range(int(self.dataList.__len__() * 0.3)):
            index = random.randint(0, len(self.dataList) - 1)
            temp = self.dataList.pop(index)  # 分出去了
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
        optim = torch.optim.SGD(self.value_net.parameters(), lr=0.0001)  # 优化器
        # loss_func = torch.nn.MSELoss()
        # loss_func = torch.nn.CrossEntropyLoss()
        loss_func = torch.nn.NLLLoss()  # Negative Log Likelihood Loss,通常用于多分类问题，是一种损失函数的计算方法
        loss1000 = 0
        # count = 0
        max_correct = 0

        loss_y = []
        correct_y = []
        epoch = []
        for step in range(1, 500001):
            index = random.randint(0, len(self.dataList) - 1)
            state = self.dataList[index].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = torch.log(self.value_net(state_tensor) + 1e-10)
            predictionRuntime = predictionRuntime.view(1, -1)

            label = []
            label.append(self.dataList[index].label)
            label_tensor = torch.tensor(label)
            # print(label_tensor)

            loss = loss_func(predictionRuntime, label_tensor)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss1000 += loss.item()
            if step % 1000 == 0:
                print('[{}]  Epoch: {}, Loss: {:.5f}'.format(datetime.now(), step, loss1000))
                # loss1000 = 0
                # self.test_network() ??
                # print('[{}]  Epoch: {}, Loss: {:.5f}'.format(datetime.now(), step, loss1000))
            if step % 1000 == 0:
                torch.save(self.value_net.state_dict(), self.args.save_dir + 'supervised_op.pt')
                loss_y.append(loss1000)
                loss1000 = 0
                correct_y.append(self.test_network())
                epoch.append(step)
                if self.test_network() > max_correct:
                    max_correct = self.test_network()
                    torch.save(self.value_net.state_dict(), self.args.save_dir + 'supervised_best_op.pt')

        # 绘制折线图
        plt.figure(figsize=(8, 5))

        plt.plot(epoch, correct_y)

        # 绘制正确率
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.title("epoch-accuracy")
        plt.savefig('accuracy_300000_op.png')
        # plt.show()

        plt.plot(epoch, loss_y)

        # 绘制损失值
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("epoch-loss")
        plt.savefig('loss_300000_op.png')
        # plt.show()

    # functions to test the network
    # 测试网络的函数
    def test_network(self):
        device = torch.device("cuda:0")
        self.load_data()
        model_path = self.args.save_dir + 'supervised_best_op.pt'
        # self.actor_net = self.value_net(self.num_inputs, self.num_output)
        self.value_net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.value_net.eval()

        correct = 0
        for step in range(self.testList.__len__()):
            state = self.testList[step].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = self.value_net(state_tensor)
            prediction = predictionRuntime.detach().cpu().numpy()
            # prediction = predictionRuntime.detach()
            maxindex = np.argmax(prediction)
            label = self.testList[step].label
            # print(maxindex, "\t", label)
            if maxindex == label:
                correct += 1
            # elif maxindex/4 == label/4:
            #     correct += 0.8
        print(correct, self.testList.__len__(), correct / self.testList.__len__(), end=' ')

        correct1 = 0
        for step in range(self.dataList.__len__()):
            state = self.dataList[step].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = self.value_net(state_tensor)
            # prediction = predictionRuntime.detach().cpu().numpy()[0]
            prediction = predictionRuntime.detach().cpu().numpy()
            maxindex = np.argmax(prediction)
            label = self.dataList[step].label
            # print(maxindex, "\t", label)
            if maxindex == label:
                correct1 += 1
            # elif maxindex/4 == label/4:
            #     correct1 += 0.8
        print(correct1, self.dataList.__len__(), correct1 / self.dataList.__len__())
        self.right = correct / self.testList.__len__()
        return correct / self.testList.__len__()

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
