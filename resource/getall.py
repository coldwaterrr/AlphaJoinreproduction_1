import os
import random
import shutil


# def random_sample(lst, k):
#     return random.sample(lst, k)
#
#
# f0 = open("sample_0")
# f1 = open("sample_1")
# f2 = open("sample_2")
# f3 = open("sample_3")
#
# sample_0 = []
# sample_1 = []
# sample_2 = []
# sample_3 = []
#
# for queryname in f0.readlines():
#     sample_0.append(queryname.split('\t')[0])
#
# for queryname in f1.readlines():
#     sample_1.append(queryname.split('\t')[0])
#
# for queryname in f2.readlines():
#     sample_2.append(queryname.split('\t')[0])
#
# for queryname in f3.readlines():
#     #print(queryname.split('\t'))
#     sample_3.append(queryname.split('\t')[0])
#
# sample_0 = random_sample(sample_0, 146 * 5)
# sample_1 = random_sample(sample_1, 146 * 3)
# sample_2 = random_sample(sample_2, 146 * 3)
# sample_3 = random_sample(sample_3, 146)
#
# sample = sample_0 + sample_1 + sample_2 + sample_3

# f0.close()
# f1.close()
# f2.close()
# f3.close()

# sample_0 = []
# sample_1 = []
# sample_2 = []
# sample_3 = []
#
# f = open('distinct_runtim_operator')
# rows = f.readlines()
# for line in rows:
#     print(line.split(',')[0])
#     print(line.split(',')[9].split('\t')[2].split(':')[0])
#     if line.split(',')[9].split('\t')[2].split(':')[0] == '0':
#         sample_0.append(line.split(',')[0])
#     elif line.split(',')[9].split('\t')[2].split(':')[0] == '1':
#         sample_1.append(line.split(',')[0])
#     elif line.split(',')[9].split('\t')[2].split(':')[0] == '2':
#         sample_2.append(line.split(',')[0])
#     else:
#         sample_3.append(line.split(',')[0])
#
# sample = sample_0 + sample_1 + sample_2 + sample_3


# print(len(sample_3))
# print(sample)

t = []

for filename in os.listdir('dataset'):
    t.append(filename)

print(len(t))

# source_folder = '../imdb_pg_dataset-master/imdb_pg_dataset-master/job_d'
# target_folder = 'dataset'
#
# os.makedirs(target_folder, exist_ok=True)
#
# for filename in os.listdir(source_folder):
#     # print(filename.split('.')[0])
#     file_path = os.path.join(source_folder, filename)
#     target_path = os.path.join(target_folder, filename)
#     if filename.split('.')[0] in sample:
#         # print(filename)
#         shutil.copy2(file_path, target_path)



# f = open()

# print(len(sample_0))
# print(sample_0)
# print(len(sample_1))
# print(sample_1)
# print(len(sample_2))
# print(sample_2)
# print(len(sample_3))
# print(sample_3)
