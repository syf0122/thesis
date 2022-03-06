import os
import glob
import sys
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
from tqdm import tqdm

# get all files
dir = sys.argv[1]
files = os.listdir(dir)
files.sort()
task = []
task_data = []
rest = []
rest_data = []
movie = []
movie_data = []
print(dir)
print(len(files))
quit()
# load data
progress_bar = tqdm(range(len(files)))
for f in files:
    if '.csv' in f:
        if 'movie' in f:
            movie.append(f)
            df = pd.read_csv(dir+f,index_col=0)
            movie_data.append(df)
        elif 'rest' in f:
            rest.append(f)
            df = pd.read_csv(dir+f,index_col=0)
            rest_data.append(df)
        elif 'ret' in f:
            task.append(f)
            df = pd.read_csv(dir+f,index_col=0)
            task_data.append(df)
    progress_bar.update()
progress_bar.close()

# calculate the average
for i in range(len(task_data)):
    if i == 0:
        task_sum = task_data[i]
    else:
        task_sum += task_data[i]
task_avg = task_sum / len(task_data)
task_avg.to_csv(dir + "avg_task_corr.csv")
hm1 = sns.heatmap(task_avg, cmap = 'jet')
hm1.set(xlabel='Vertex', ylabel='Latent', title = "Averaged Task Correlation Matrix")
plt.savefig(dir+"avg_task_corr.jpg", bbox_inches='tight')
plt.clf()

for i in range(len(rest_data)):
    if i == 0:
        rest_sum = rest_data[i]
    else:
        rest_sum += rest_data[i]
rest_avg = rest_sum / len(rest_data)
rest_avg.to_csv(dir + "avg_rest_corr.csv")
hm1 = sns.heatmap(rest_avg, cmap = 'jet')
hm1.set(xlabel='Vertex', ylabel='Latent', title = "Averaged Rest Correlation Matrix")
plt.savefig(dir+"avg_rest_corr.jpg", bbox_inches='tight')
plt.clf()

for i in range(len(movie_data)):
    if i == 0:
        movie_sum = movie_data[i]
    else:
        movie_sum += movie_data[i]
movie_avg = movie_sum / len(movie_data)
movie_avg.to_csv(dir + "avg_movie_corr.csv")
hm1 = sns.heatmap(movie_avg, cmap = 'jet')
hm1.set(xlabel='Vertex', ylabel='Latent', title = "Averaged movie Correlation Matrix")
plt.savefig(dir+"avg_movie_corr.jpg", bbox_inches='tight')
plt.clf()
