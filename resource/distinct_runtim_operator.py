import os

f = open("runtime_operator")
runtime = f.readlines()
distinct_hint = []
f1 = open("distinct_runtim_operator","w+")


for line in runtime:
    hint = line.split(',')[1]
    if(hint not in distinct_hint):
        f1.write(line)
        # f1.write('\n')
        distinct_hint.append(hint)
f1.close()
print(distinct_hint)
    # print(line)