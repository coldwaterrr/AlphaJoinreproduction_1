# The original table name is sequentially transferred to the corresponding queryName file
import os
import psycopg2

import numpy as np

querydir = '../resource/jobquery'  # JOB query
tablenamedir = '../resource/jobtablename'  # tablename involved in the query statement  查询中的表名
shorttolongpath = '../resource/shorttolong'  # Mapping of table abbreviations to full names 表缩写到全名的映射

def getResource():
    conn = psycopg2.connect(database="imdb", user="postgres", password="postgres", host="localhost", port="5432")
    cur = conn.cursor()

    short_to_long = {}

    fileList = os.listdir(querydir)
    fileList.sort()
    for queryName in fileList:
        # queryName = "9d.sql"
        querypath = querydir + "/" + queryName
        file_object = open(querypath)
        file_context = file_object.readlines()  # 可以一次性读取文件所有数据 ,返回结果是一个列表 ,列表中的每个元素对应文件中的一行元素
        # print(file_context)
        file_object.close()

        j = 0
        k = 0
        tablenames = []
        for i in range(len(file_context)):  # file_context是个列表
            if file_context[i].find("FROM") != -1:
                break
            j = j + 1
        for i in range(len(file_context)):
            if file_context[i].find("WHERE") != -1:
                break
            k = k + 1

        # print(j,end="\nk=")
        # print(k)
        # The original table name is sequentially transferred to the corresponding queryName file, update tablename
        # 原始表名按顺序传输到相应的queryName文件，更新表名(将表名的别名放进tablenames)
        for i in range(j, k - 1):
            temp = file_context[i].split()
            tablenames.append(temp[temp.index("AS") + 1][:-1].lower())
        temp = file_context[k - 1].split()
        tablenames.append(temp[temp.index("AS") + 1].lower())

        # print(tablenames)

        f = open(tablenamedir + "/" + queryName[:-4], 'w')
        f.write(str(tablenames))
        f.close()
        # print(queryName, tablenames)

        # Read query
        querypath = querydir + "/" + queryName
        file_object = open(querypath)
        file_context = file_object.read()
        file_object.close()

        # print(file_context)

        # 获取查询计划
        cur.execute("explain " + file_context)
        rows = cur.fetchall()  # 获取得到的所有结果
        # print(row)

        scan_language = []

        for line in rows:
            if line[0].find('Scan') != -1 & line[0].find('Bitmap Index') == -1:
                scan_language.append(line[0])
        for language in scan_language:
            word = language.split(' ')
            index = word.index('on')
            short_to_long[word[index + 2]] = word[index + 1]

    print(len(short_to_long))

    # Dump two dict to corresponding files
    f = open(shorttolongpath, 'w')
    f.write(str(short_to_long))
    f.close()

    cur.close()
    conn.close()


if __name__ == '__main__':
    getResource()