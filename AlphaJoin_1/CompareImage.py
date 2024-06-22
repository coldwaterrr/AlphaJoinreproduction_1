import os
import matplotlib.pyplot as plt
import numpy as np
import random
import math

f_original = open("result_original")
f_k4 = open("result_k4")
f_k6 = open("result_k6")
f_k5 = open("result_k5")

line_original = f_original.readlines()
line_k4 = f_k4.readlines()
line_k6 = f_k6.readlines()
line_k5 = f_k5.readlines()

f_original.close()
f_k4.close()
f_k6.close()
f_k5.close()

runtime_original_dict = {}
runtime_k4_dict = {}
runtime_k6_dict = {}
runtime_k5_dict = {}
queryname = []
runtime_original = []
runtime_k4 = []
runtime_k6 = []
runtime_k5 = []
flag = 0

hint_length = set()

for line in line_original:
    # if flag == 20:
    #     break
    # flag = flag + 1
    # hint_length.add(len(line.split(",")[1].strip()))
    if len(line.split(",")[1].strip()) < 80:
        queryname.append(line.split(",")[0].strip())


print(len(queryname))


# queryname = random.sample(queryname, 50)


for line in line_original:
    # if flag == 20:
    #     break
    # flag = flag + 1
    # queryname.append(line.split(",")[0].strip())
    if line.split(",")[0].strip() in queryname and len(line.split(",")[1].strip()) < 80:
        print(line.split(",")[0].strip() + ":", end=" ")
        print(len(line.split(",")[1].strip()))
        # runtime_original.append(float(line.split(",")[2].strip()))
        runtime_original_dict[line.split(",")[0].strip()] = float(line.split(",")[2].strip())

flag = 0

for line in line_k4:
    # if flag == 20:
    #     break
    # flag = flag + 1
    if line.split(",")[0].strip() in queryname and len(line.split(",")[1].strip()) < 80:
        # runtime_k4_3.append(float(line.split(",")[2].strip()))
        runtime_k4_dict[line.split(",")[0].strip()] = float(line.split(",")[2].strip())

for line in line_k5:
    # if flag == 20:
    #     break
    # flag = flag + 1
    if line.split(",")[0].strip() in queryname and len(line.split(",")[1].strip()) < 80:
        # runtime_k4_3.append(float(line.split(",")[2].strip()))
        runtime_k5_dict[line.split(",")[0].strip()] = float(line.split(",")[2].strip())

for line in line_k6:
    # if flag == 20:
    #     break
    # flag = flag + 1
    if line.split(",")[0].strip() in queryname and len(line.split(",")[1].strip()) < 80:
        # runtime_k4_3.append(float(line.split(",")[2].strip()))
        runtime_k6_dict[line.split(",")[0].strip()] = float(line.split(",")[2].strip())

for query in queryname:
    runtime_original.append(math.sqrt(math.sqrt(runtime_original_dict[query])))
    runtime_k4.append(math.sqrt(math.sqrt(runtime_k4_dict[query])))
    runtime_k6.append(math.sqrt(math.sqrt(runtime_k6_dict[query])))
    runtime_k5.append(math.sqrt(math.sqrt(runtime_k5_dict[query])))

print(runtime_k4)
print(runtime_k5)
print(runtime_k6)

correct1 = 0
correct2 = 0
correct3 = 0

print("4标签:")

for i in range(len(runtime_k4)):
    if runtime_k4[i] < runtime_original[i]:
        correct1 = correct1 + 1
    else:
        print(queryname[i], end=' ')

print("")
print("5标签:")

for i in range(len(runtime_k6)):
    if runtime_k6[i] < runtime_original[i]:
        correct2 = correct2 + 1
    else:
        print(queryname[i], end=' ')

print("")
print("6标签:")

for i in range(len(runtime_k5)):
    if runtime_k5[i] < runtime_original[i]:
        correct3 = correct3 + 1
    else:
        print(queryname[i], end=' ')

print(len(queryname))
print(sum(runtime_original))
print("4标签:", correct1, '\t', correct1/len(queryname), '\t', sum(runtime_k4))
print("6标签:", correct2, '\t', correct2/len(queryname), '\t', sum(runtime_k6))
print("5标签:", correct3, '\t', correct3/len(queryname), '\t', sum(runtime_k5))
# print(queryname)
# print(runtime_original)
# print(runtime_k4_3)


correct = [correct1, correct3, correct2]
correct_rate = [correct1/len(queryname), correct3/len(queryname), correct2/len(queryname)]

x = ["k4", "k5", "k6"]
y1 = correct
y2 = correct_rate

# 创建一个图形
fig, ax1 = plt.subplots()

# 绘制第一个数组的直方图
color = 'tab:blue'
bar_width = 0.2  # 设置条形图的宽度
ax1.set_xlabel('Number of tags')
ax1.set_ylabel('Correct', color=color)
ax1.bar(x, correct, color=color, alpha=0.5, width=bar_width)
ax1.tick_params(axis='y', labelcolor=color)

# 创建一个共享x轴的第二个y轴
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Correct Rate', color=color)
ax2.plot(x, correct_rate, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

# 添加标题
plt.title('Correct and Correct Rate')

# 显示图形
plt.show()



# 参数设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 20})  # 设置字体大小


# 将横坐标班级先替换为数值
x = np.arange(len(queryname))
width = 0.2
k4 = x
k5 = x + width
k6 = x + 2 * width
original = x + 3 * width

# 绘图
plt.bar(k4, runtime_k4, width=width, color='gold', label='4标签')
plt.bar(k5, runtime_k5, width=width, color='b', label='5标签')
plt.bar(k6, runtime_k6, width=width, color="c", label="6标签")
plt.bar(original, runtime_original, width=width, color="silver", label="原始数据")


# 设置初始横坐标位置，为了简化计算，我们不直接基于班级数量计算x，而是手动控制每个条形的位置
num_bars = 4  # 总共四组条形图（4标签、5标签、6标签、原始数据）
bar_width = 0.2  # 条形宽度
x_ticks_pos = x + (bar_width * num_bars / 2) - bar_width * (num_bars // 2 - 0.5)  # 计算刻度标签的中心位置

# 将横坐标数值转换为班级
plt.xticks(x_ticks_pos, labels=queryname)

# 显示柱状图的高度文本
for i in range(len(queryname)):
    plt.text(k4[i], runtime_k4[i], runtime_k4[i], va="bottom", ha="center", fontsize=5)
    plt.text(k5[i], runtime_k5[i], runtime_k5[i], va="bottom", ha="center", fontsize=5)
    plt.text(k6[i], runtime_k6[i], runtime_k6[i], va="bottom", ha="center", fontsize=5)
    plt.text(original[i], runtime_original[i], runtime_original[i], va="bottom", ha="center", fontsize=5)

# 显示图例
plt.legend(loc="upper right")
plt.savefig("柱形图", dpi=1000)
plt.show()


# print(runtime_original)