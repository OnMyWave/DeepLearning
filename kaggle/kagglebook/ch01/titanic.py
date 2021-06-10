import numpy as np
import pandas as pd
import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(os.path.dirname("/Users/onmywave/Desktop/Github/DeepLearning/kaggle/kagglebook-main/input/ch01-titanic/"))

train = pd.read_csv("train.csv")
test = pd.read_csv("test.cvs")

print(train)