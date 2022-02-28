import os
import glob
import sys
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib

# get all task files
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
plt.clf()


# get all rest files
files = os.listdir("rest/")
files.sort()
rest = []
rest_data = []

for f in files:
    if ".csv" in f:
        rest.append(f)
        df = pd.read_csv("rest/"+f,index_col=0)
        rest_data.append(df)

for i in range(len(rest_data)):
    if i == 0:
        rest_sum = rest_data[i]
    else:
        rest_sum += rest_data[i]

rest_avg = rest_sum / len(rest_data)
rest_avg.to_csv("average/rest_corr_avg.csv")
hm1 = sns.heatmap(rest_avg, cmap = 'jet')
hm1.set(xlabel='Vertex', ylabel='Latent', title = "Averaged Rest Correlation Matrix")
plt.savefig("average/rest_avg_corr.jpg", bbox_inches='tight')
plt.clf()


# get all movie files
files = os.listdir("movie/")
files.sort()
movie = []
movie_data = []

for f in files:
    if ".csv" in f:
        movie.append(f)
        df = pd.read_csv("movie/"+f,index_col=0)
        movie_data.append(df)

for i in range(len(movie_data)):
    if i == 0:
        movie_sum = movie_data[i]
    else:
        movie_sum += movie_data[i]

movie_avg = movie_sum / len(movie_data)
movie_avg.to_csv("average/movie_corr_avg.csv")
hm1 = sns.heatmap(movie_avg, cmap = 'jet')
hm1.set(xlabel='Vertex', ylabel='Latent', title = "Averaged movie Correlation Matrix")
plt.savefig("average/movie_avg_corr.jpg", bbox_inches='tight')
plt.clf()
