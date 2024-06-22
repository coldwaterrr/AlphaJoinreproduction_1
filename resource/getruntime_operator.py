import os

f = open("distinct_runtim_operator")

old = f.readlines()

f1 = open("runtime_op", "w+")

count = 0

for line in old:
    print(line.split(',')[0])
    print(line.split(',')[1])
    print(line.split(',')[9].split('\t')[2].split(':')[1].replace('\n', ''))
    print(line.split(',')[9].split('\t')[2].split(':')[0])
    f1.write(line.split(',')[0]+',')
    f1.write(line.split(',')[1]+',')
    f1.write(line.split(',')[9].split('\t')[2].split(':')[1].replace('\n', '')+',')
    f1.write(line.split(',')[9].split('\t')[2].split(':')[0]+'\n')
    count = count + 1
f.close()
f1.close()
print(count)