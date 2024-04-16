import os
import json
import ast
from getResource import getResource

querydir = '../resource/jobquery'  # imdb的查询语句
tablenamedir = '../resource/jobtablename'  # 每条语句对应表名的别名(缩写)
shorttolongpath = '../resource/shorttolong'  # 缩写与全名的映射
predicatesEncodeDictpath = './predicatesEncodedDict'
queryEncodeDictpath = './queryEncodedDict'

# 打开文件并读取内容
with open('shottolong.txt', 'r') as file:
    data = file.readlines()
# print(type(data))
shorttolong = dict()
for i in data:
    if i == '\n' or i =='Process finished with exit code 0\n':
        continue
    # print(i)
    i.strip('n')
    i.strip('\\')
    print(i.split(':')[1])
    shorttolong[i.split(':')[0].replace("'", "").strip(' ')] = i.split(':')[1].replace("'", "").strip(' ').replace("\n","")
# print(shorttolong['mc'])
# print(shorttolong)
# print(data)
# shorttolong = ast.literal_eval(data)
# longtoshort = json.loads(data)
# longtoshort.txt = dict(longtoshort.txt)
# print(type(data))
# print(longtoshort)

# Get all the attributes used to select the filter vector
# 获取用于选择滤波器向量的所有属性
def getQueryAttributions():
    fileList = os.listdir(querydir)
    fileList.sort()
    attr = set()  # 创建一个无序不重复的元素集
    selectivity = dict()



    for queryName in fileList:
        querypath = querydir + "/" + queryName
        file_object = open(querypath)
        file_context = file_object.readlines()  # 获取query语句
        file_object.close()

        # find WHERE
        k = -1
        for i in range(len(file_context)):
            k = k + 1
            if file_context[i].find("WHERE") != -1:
                break

        # handle a sentence after WHERE
        # 处理 WHERE 后的句子
        for i in range(k, len(file_context)):
            temp = file_context[i].split()  # 默认空格分隔
            for word in temp:
                if '.' in word:
                    if word[0] == "'":
                        continue
                    if word[0] == '(':
                        word = word[1:]  # object[start:end:step]   object[:]表示从头取到尾，步长默认为1  object[::]一样表示从头到尾，步长为1
                    if word[-1] == ';':  # object[:5]没有Start表示从头开始取,步长为1，object[5:]表示从5开始到尾，步长为1
                        word = word[:-1]

                    short_tablename = word.split('.')[0]
                    column = word.split('.')[1]
                    # print(type(column))
                    long_tablename = shorttolong[short_tablename]
                    print(long_tablename)
                    # selectivity[word] = s_value

                    key_exist = long_tablename in selectivity
                    if key_exist == False:
                        
                    else:


                    # print(word)
                    attr.add(word)

    attrNames = list(attr)
    attrNames.sort()
    return attrNames


def getQueryEncode(attrNames):
    print(attrNames)
    print(len(attrNames))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Read all table abbreviations
    # 读取所有表格缩写
    f = open(shorttolongpath, 'r')
    a = f.read()  # 读取文件的整个内容,返回的是字符串
    short_to_long = eval(a)  # 的功能是去掉参数最外侧引号，变成python可执行的语句，并执行语句的函数。
    f.close()
    tableNames = []

    for i in short_to_long:
        tableNames.append(i)
    tableNames.sort()

    # Mapping of table name abbreviations and numbers (list subscripts)
    # 表名缩写和编号（列表下标）的映射
    table_to_int = {}
    int_to_table = {}
    for i in range(len(tableNames)):
        int_to_table[i] = tableNames[i]
        table_to_int[tableNames[i]] = i

    # Mapping of attributes and numbers (list subscripts)
    # 属性和编号（列表下标）的映射
    attr_to_int = {}
    int_to_attr = {}
    for i in range(len(attrNames)):
        int_to_attr[i] = attrNames[i]
        attr_to_int[attrNames[i]] = i
    # print(table_to_int)

    queryEncodeDict = {}
    joinEncodeDict = {}
    predicatesEncodeDict = {}
    fileList = os.listdir(querydir)
    fileList.sort()

    for queryName in fileList:
        joinEncode = [0 for _ in range(len(tableNames) * len(tableNames))]
        predicatesEncode = [0 for _ in range(len(attrNames))]

        # Read query statement
        # 读取查询语句
        querypath = querydir + "/" + queryName
        file_object = open(querypath)
        file_context = file_object.readlines()
        file_object.close()

        # find WHERE
        k = -1
        for i in range(len(file_context)):
            k = k + 1
            if file_context[i].find("WHERE") != -1:
                break

        # handle a sentence after WHERE
        for i in range(k, len(file_context)):
            temp = file_context[i].split()

            if "=" in temp:
                index = temp.index("=")
                if (filter(temp[index - 1]) in attrNames) & (filter(temp[index + 1]) in attrNames):
                    table1 = temp[index - 1].split('.')[0]
                    table2 = temp[index + 1].split('.')[0]
                    joinEncode[table_to_int[table1] * len(tableNames) + table_to_int[table2]] = 1
                    joinEncode[table_to_int[table2] * len(tableNames) + table_to_int[table1]] = 1
                else:
                    for word in temp:
                        if '.' in word:
                            if word[0] == "'":
                                continue
                            if word[0] == '(':
                                word = word[1:]
                            if word[-1] == ';':
                                word = word[:-1]
                            predicatesEncode[attr_to_int[word]] = 1
            else:
                for word in temp:
                    if '.' in word:
                        if word[0] == "'":
                            continue
                        if word[0] == '(':
                            word = word[1:]
                        if word[-1] == ';':
                            word = word[:-1]
                        predicatesEncode[attr_to_int[word]] = 1
        predicatesEncodeDict[queryName[:-4]] = predicatesEncode
        queryEncodeDict[queryName[:-4]] = joinEncode + predicatesEncode

    for i in queryEncodeDict.items():
        print(i)
    # print(len(tableNames), tableNames)
    # print(len(attrNames), attrNames)

    f = open(predicatesEncodeDictpath, 'w')
    f.write(str(predicatesEncodeDict))
    f.close()
    f = open(queryEncodeDictpath, 'w')
    f.write(str(queryEncodeDict))
    f.close()


def filter(word):
    if '.' in word:
        if word[0] == '(':
            word = word[1:]
        if word[-1] == ';':
            word = word[:-1]
    return word


if __name__ == '__main__':
    getResource()
    attrNames = getQueryAttributions()
    getQueryEncode(attrNames)
