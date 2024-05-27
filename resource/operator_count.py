import os

f = open('distinct_runtim_operator')
# f1 = open('sample_0', "w+")


rows = f.readlines()
operator_count = [0, 0, 0, 0, 0, 0, 0, 0]

for line in rows:
    operator_count[int(line.split(',')[9].split('\t')[2].split(':')[0])] = operator_count[int(line.split(',')[9].split('\t')[2].split(':')[0])] + 1
    print(line.split(',')[0], end=":")
    print(line.split(',')[9].split('\t')[2].split(':')[0])
    # if(int(line.split(',')[9].split('\t')[2].split(':')[0]) == 0):
    #     f1.write(line.split(',')[0])
    #     f1.write("\t")
    #     f1.write(line.split(',')[9].split('\t')[2].split(':')[0])
    #     f1.write('\n')


print(operator_count)  # [18484, 631, 739, 146, 0, 0, 0, 0]
# [402, 40, 131, 42, 0, 0, 0, 0]