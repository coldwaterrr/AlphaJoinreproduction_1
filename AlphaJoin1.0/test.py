from datetime import datetime
import random
import os
import numpy as np
import torch
import pickle
from models import ValueNet

# line = "Finalize Aggregate  (cost=533822.72..533822.73 rows=1 width=64)"
#
# queryName = line.split(",")[0].encode('utf-8').decode('utf-8-sig').strip()
# hint = line.split(",")[1]
# runtime = line.split(",")[2].strip()
#
# print(queryName)
# print(hint)
# print(runtime)

hint = "( ( ( ( ( mc ( ci rt ) ) cn ) t ) chn ) ct )"
tablesInQuery = hint.split(" ")
print("tablesInQuery:", end="")
print(tablesInQuery)
stack = []
difference = 0
for i in tablesInQuery:
    if i == ')':
        tempb = stack.pop()
        tempa = stack.pop()
        print("tempb:" + tempb)
        print("tempa:" + tempa)
        _ = stack.pop()
        b = tempb.split('+')
        a = tempa.split('+')
        print("a:", end="")
        print(a)
        print("b:", end="")
        print(b)
        b.sort()
        a.sort()
        print("a.sort():", end="")
        print(a)
        print("b.sort():", end="")
        print(b)
        # indexb = self.table_to_int[b[0]]
        # indexa = self.table_to_int[a[0]]
        # matrix[indexa, indexb] = (len(tablesInQuery) + 2) / 3 - difference
        difference += 1
        stack.append(tempa + '+' + tempb)
        print("if:\t", end="")
        print(stack)
        print("\n")
    else:
        print("else:\t", end="")
        stack.append(i)
        print(stack)
        print("\n")

print("final:\t")
print(stack)
