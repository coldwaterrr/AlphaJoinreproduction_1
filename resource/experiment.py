import os

f = open("experiment")

rows = f.readlines()

origin = []
new = []

for row in rows:
    print(row.split(" ")[1])
    print(row.split(" ")[4].split("..")[0].split("=")[1])
    origin.append(row.split(" ")[1])
    new.append(row.split(" ")[4].split("..")[0].split("=")[1])
    # print(row.split(" ")[1])
    # print(row.split(" ")[2])
f.close()

