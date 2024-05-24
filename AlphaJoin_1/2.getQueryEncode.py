import os
import json
import ast
from getResource import getResource
import psycopg2

querydir = '../resource/dataset'  # imdb的查询语句
tablenamedir = '../resource/jobtablename'  # 每条语句对应表名的别名(缩写)
shorttolongpath = '../resource/shorttolong'  # 缩写与全名的映射
predicatesEncodeDictpath = './predicatesEncodedDict'
queryEncodeDictpath = './queryEncodedDict'

# 数据库连接参数
print("connecting...")
conn = psycopg2.connect(database="imdb", user="postgres", password="postgres", host="localhost", port="5432")
cur = conn.cursor()
print("connect success")

# 打开文件并读取内容
with open('shottolong.txt', 'r') as file:
    data = file.readlines()
# print(type(data))

key_list = []

shorttolong = dict()
for i in data:
    if i == '\n' or i =='Process finished with exit code 0\n':
        continue
    # print(i)
    i.strip('n')
    i.strip('\\')
    # print(i.split(':')[1])
    print(i.split(':')[0].replace("'", "").strip(' '))
    key_list.append(i.split(':')[0].replace("'", "").strip(' '))
    shorttolong[i.split(':')[0].replace("'", "").strip(' ')] = i.split(':')[1].replace("'", "").strip(' ').replace("\n","")
# print(shorttolong['mc'])
# print(shorttolong)
# print(data)
# shorttolong = ast.literal_eval(data)
# longtoshort = json.loads(data)
# longtoshort.txt = dict(longtoshort.txt)
# print(type(data))
# print(longtoshort)

selectivity = dict()

# Get all the attributes used to select the filter vector
# 获取用于选择滤波器向量的所有属性
def getQueryAttributions():
    fileList = os.listdir(querydir)
    fileList.sort()
    attr = set()  # 创建一个无序不重复的元素集

    rowscount = dict()



    for queryName in fileList:
        print(queryName)
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

                    if word.split('.')[0] not in key_list:
                        continue

                    short_tablename = word.split('.')[0]
                    column = word.split('.')[1]
                    # print(type(column))
                    print(short_tablename)
                    long_tablename = shorttolong[short_tablename]
                    # print(long_tablename)
                    # selectivity[word] = s_value

                    if column == 'kind' or column == 'id' or column == 'role':
                        selectivity[word] = 1.0

                    # 最简单的选择率:唯一值数/行数
                    # 获取每一张表的具体行数
                    key_exist_1 = word in selectivity
                    key_exist_2 = long_tablename in rowscount
                    if key_exist_1 == False and key_exist_2 == False:
                        sql = '''
                        select count(*) from %s
                        '''% (long_tablename)
                        cur.execute(sql)
                        rows = cur.fetchall()
                        # print(type(rows))
                        # print(rows)
                        for row in rows:
                            rowscount[long_tablename] = float(row[0])

                    # 查出唯一值，计算选择率
                        sql='''
                        select n_distinct from pg_stats where tablename = '%s' and attname = '%s'
                        ''' % (long_tablename, column)
                        cur.execute(sql)
                        rows = cur.fetchall()
                        # print(rows)
                        for row in rows:
                            n_distinct = float(row[0])
                            # if n_distinct > rowscount[long_tablename]:
                            #     print(column+'  n_distinct:', end='')
                            #     print(float(row[0]))
                            #     print('rowscount  '+long_tablename+':', end='')
                            #     print(rowscount[long_tablename])
                        if n_distinct < 0:
                            selectivity[word] = -n_distinct
                        else:
                            selectivity[word] = n_distinct / rowscount[long_tablename]
                        # for row in rows:
                        #     if int(row) < 0:


                    elif key_exist_1 == False and key_exist_2 == True:
                        # 查出唯一值，计算选择率
                        sql = '''
                        select n_distinct from pg_stats where tablename = '%s' and attname = '%s'
                        ''' % (long_tablename, column)
                        cur.execute(sql)
                        rows = cur.fetchall()
                        # print(rows)
                        for row in rows:
                            n_distinct = float(row[0])
                            # if n_distinct > rowscount[long_tablename]:
                            #     print(column + '  n_distinct:', end='')
                            #     print(float(row[0]))
                            #     print('rowscount  ' + long_tablename + ':', end='')
                            #     print(rowscount[long_tablename])
                        if n_distinct < 0:
                            selectivity[word] = -n_distinct
                        else:
                            selectivity[word] = n_distinct / rowscount[long_tablename];

                        # if n_distinct > float(rowscount[long_tablename]):
                        #     print(column + '  n_distinct:', end='')
                        #     print(float(row[0]))
                        #     print('rowscount  ' + long_tablename + ':', end='')
                        #     print(float(rowscount[long_tablename]))
                        if n_distinct < 0:
                            selectivity[word] = -n_distinct
                        else:
                            selectivity[word] = n_distinct / float(rowscount[long_tablename])

                    # print(word)
                    attr.add(word)

    attrNames = list(attr)
    attrNames.sort()
    return attrNames


def getQueryEncode(attrNames):
    # print(attrNames)
    # print(len(attrNames))
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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
    # print('attr_to_int')
    # print(attr_to_int)
    queryEncodeDict = {}
    joinEncodeDict = {}
    predicatesEncodeDict = {}
    fileList = os.listdir(querydir)
    fileList.sort()

    # 算子编码预处理
    operator = {}
    operator_path = '../resource/distinct_runtim_operator'
    f = open(operator_path)
    rows = f.readlines()
    for line in rows:
        operator[line.split(',')[0]] = line.split(',')[9].split('\t')[2].split(':')[0]
    f.close()
    print(operator)

    for queryName in fileList:
        joinEncode = [0 for _ in range(len(tableNames) * len(tableNames))]
        predicatesEncode = [0 for _ in range(len(attrNames))]

        # 算子编码信息
        if operator[queryName.split('.')[0]] == '0':
            op = [0, 0, 0]
        elif operator[queryName.split('.')[0]] == '1':
            op = [0, 0, 1]
        elif operator[queryName.split('.')[0]] == '2':
            op = [0, 1, 0]
        elif operator[queryName.split('.')[0]] == '3':
            op = [0, 1, 1]

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
                            if word.split('.')[0] not in key_list:
                                continue
                            predicatesEncode[attr_to_int[word]] = selectivity[word]
                            # predicatesEncode = predicatesEncode
            else:
                for word in temp:
                    if '.' in word:
                        if word[0] == "'":
                            continue
                        if word[0] == '(':
                            word = word[1:]
                        if word[-1] == ';':
                            word = word[:-1]
                        if word.split('.')[0] not in key_list:
                            continue
                        predicatesEncode[attr_to_int[word]] = selectivity[word]
        predicatesEncodeDict[queryName[:-4]] = predicatesEncode + op
        queryEncodeDict[queryName[:-4]] = joinEncode + predicatesEncode + op

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
