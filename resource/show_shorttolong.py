import os
import pandas as pd


f1 = open("shorttolong")
# for line in f1.read():
#     print(line)
# print(type(f1.read()))

stt = f1.read()
# print(stt)
stt = stt.strip('{').strip('}')
# print(stt)
stt = stt.split(',')
# print(stt)

for line in stt:
    print(line)

