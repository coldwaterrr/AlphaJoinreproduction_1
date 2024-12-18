import os
import psycopg2
import random

# querydir = '../resource/jobquery'  # JOB query
# querydir = '../imdb_pg_dataset-master/imdb_pg_dataset-master/job_extended'
querydir = '../imdb_pg_dataset-master/imdb_pg_dataset-master/job_d'

conn = psycopg2.connect(database="imdb", user="postgres", password="postgres", host="localhost", port="5432")
cur = conn.cursor()

def simple_random_sampling(lst, k):
    return random.sample(lst, k)

fileList = os.listdir(querydir)
fileList.sort()

sample = simple_random_sampling(fileList, 500)
print(sample)

f1 = open("runtime_operator", "w+")

button = ['on', 'off']
query_count = 0

def getHint(queryplan, begin, end):
    if queryplan[begin].find('Scan') != -1 & queryplan[begin].find('Bitmap Index') == -1:
        language = queryplan[begin]
        word = language.split(' ')
        index = word.index('on')
        # print(word[index + 2], end="\t")
        return word[index + 2]
    if begin == end:
        return

    bla = blank(queryplan[begin])
    count = 0
    for i in range(begin + 1, end):
        if blank(queryplan[i]) == bla + 6:
            count += 1
            if count == 2:
                a = getHint(queryplan, begin + 1, i)
                b = getHint(queryplan, i, end)
                return "( " + a + " " + b + " )"

    return getHint(queryplan, begin + 1, end)

def blank(line):
    for i in range(len(line)):
        # print(line[i], end=" ")
        if line[i] == '-':
            # print(i)
            return i
    return -1

operator_count = [0,0,0,0,0,0,0,0,0]

for queryName in fileList:
    # if query_count == 200:
    #     break
    print(queryName)
    # Read query
    querypath = querydir + "/" + queryName
    file_object = open(querypath)
    file_context = file_object.read()
    file_object.close()

    count = 0
    min_cost = 10000000.0
    min_count = 0
    # print(file_context)
    # cur.execute(file_context)
    flag = 0
    op = []

    for i in button:
        # if (queryName == 'e12a.sql'):
        #     print(rows)
        for j in button:
            for k in button:
                r = 0
                cur.execute("set enable_nestloop to " + i)
                cur.execute("set enable_mergejoin to " + j)
                cur.execute("set enable_hashjoin to " + k)
                # 获取查询计划
                cur.execute("explain " + file_context)
                rows = cur.fetchall()  # 获取得到的所有结果

                queryplan = []
                for line in rows:
                    # if (queryName == 'e12a.sql'):
                    #     print(line)
                    queryplan.append(line[0])
                    if line[0].find('Aggregate') != -1 & line[0].find('Finalize') != -1:
                        costline = line[0]
                        costline = costline.split(' ')
                        # print(line[0])
                        # print(queryName + ":", end=" ")
                        # print(costline[3].split('=')[1].split('..')[0])
                        cost = costline[3].split('=')[1].split('..')[0]
                    elif line[0].find('Aggregate') != -1 & line[0].find('Finalize') == -1 & line[0].find(
                            'Partial') == -1 & line[0].find('GroupAggregate') == -1:
                        costline = line[0]
                        costline = costline.split(' ')
                        # print(line[0])
                        # print(queryName + ":", end=" ")
                        # print(costline[2].split('=')[1].split('..')[0])
                        # if(queryName == 'e12a.sql'):
                        #     print(costline)
                        #     print(costline[2])
                        cost = costline[2].split('=')[1].split('..')[0]
                    elif r == 0 & line[0].find('Sort') != -1 & line[0].find('Key') == -1:
                        costline = line[0]
                        costline = costline.split(' ')
                        print(queryName)
                        print(costline[2].split('=')[1].split('..')[0])
                        cost = costline[2].split('=')[1].split('..')[0]
                        # print(cost)
                    r = r + 1
                if flag == 0:
                    f1.write(queryName.split('.')[0])
                    f1.write(',')
                    f1.write(getHint(queryplan, 0, len(queryplan)))
                    flag = flag + 1
                f1.write(',')
                f1.write(str(count))
                f1.write(':')
                f1.write(str(cost))
                if float(cost) < float(min_cost):
                    min_cost = cost
                    min_count = count
                    if min_count != 0:
                        op.append(i)
                        op.append(j)
                        op.append(k)
                count = count + 1

    f1.write("\t\t")
    f1.write(str(min_count))
    operator_count[min_count] = operator_count[min_count] + 1
    f1.write(':')
    f1.write(str(min_cost))
    if len(op) == 3:
        f1.write(',')
        f1.write(op[0]+' '+op[1]+' '+op[2])
    f1.write('\n')
    query_count = query_count + 1

f1.close()
cur.close()
conn.close()
print(operator_count)