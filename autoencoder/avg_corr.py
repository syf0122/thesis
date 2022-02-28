import os
import glob
import sys
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib

# get all files
# get all files
files = os.listdir("task/")
files.sort()
task = []
task_data = []

for f in files:
    if ".csv" in f:
        task.append(f)
        df = pd.read_csv("task/"+f,index_col=0)
        task_data.append(df)

for i in range(len(task_data)):
    if i == 0:
        task_sum = task_data[i]
    else:
        task_sum += task_data[i]

task_avg = task_sum / len(task_data)
task_avg.to_csv("average/task_corr_avg.csv")
hm1 = sns.heatmap(task_avg, cmap = 'jet')
hm1.set(xlabel='Vertex', ylabel='Latent', title = "Averaged Task Correlation Matrix")
plt.savefig("average/task_avg_corr.jpg", bbox_inches='tight')
