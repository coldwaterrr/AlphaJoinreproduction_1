from datetime import datetime
import random
import os
import numpy as np
import torch
import pickle
from models import ValueNet

line = "Finalize Aggregate  (cost=533822.72..533822.73 rows=1 width=64)"

queryName = line.split(",")[0].encode('utf-8').decode('utf-8-sig').strip()
hint = line.split(",")[1]
runtime = line.split(",")[2].strip()

print(queryName)
print(hint)
print(runtime)