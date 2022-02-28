import os
import glob
import sys
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import hcp_utils as hcp
from nilearn import plotting as nlp
from nilearn import surface as nsf

subs = ["100610", "102311", "102816", "104416", "105923",
        "108323", "109123", "111514", "114823", "115017",
        "115825", "116726", "118225", "125525", "126426",
        "128935", "130114", "130518", "131217", "131722",
        "132118", "134627", "134829", "135124", "137128",
        "140117", "144226", "145834", "146129", "146432",
        "146735", "146937", "148133", "150423", "155938",
        "156334", "157336", "158035", "158136", "159239",
        "162935", "164131", "164636", "165436", "167036",
        "167440", "169040", "169343", "169444", "169747"]

# movie plots
for sub in subs:
    df1 = pd.read_csv("movie/"+sub+"_movie1_corr.csv",index_col=0)
    df2 = pd.read_csv("movie/"+sub+"_movie2_corr.csv",index_col=0)
    df3 = pd.read_csv("movie/"+sub+"_movie3_corr.csv",index_col=0)
    df4 = pd.read_csv("movie/"+sub+"_movie4_corr.csv",index_col=0)
    hm1 = sns.heatmap(df1, cmap = 'jet')
    hm1.set(xlabel='Vertex', ylabel='Latent', title = sub+" Movie1 Correlation Matrix")
    plt.savefig("movie_plots/"+sub+"_movie1_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm2 = sns.heatmap(df2, cmap = 'jet')
    hm2.set(xlabel='Vertex', ylabel='Latent', title = sub+" Movie2 Correlation Matrix")
    plt.savefig("movie_plots/"+sub+"_movie2_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm3 = sns.heatmap(df3, cmap = 'jet')
    hm3.set(xlabel='Vertex', ylabel='Latent', title = sub+" Movie3 Correlation Matrix")
    plt.savefig("movie_plots/"+sub+"_movie3_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm4 = sns.heatmap(df4, cmap = 'jet')
    hm4.set(xlabel='Vertex', ylabel='Latent', title = sub+" Movie4 Correlation Matrix")
    plt.savefig("movie_plots/"+sub+"_movie4_corr.jpg", bbox_inches='tight')
    plt.clf()

# rest
for sub in subs:
    df1 = pd.read_csv("rest/"+sub+"_rest1_corr.csv",index_col=0)
    df2 = pd.read_csv("rest/"+sub+"_rest2_corr.csv",index_col=0)
    df3 = pd.read_csv("rest/"+sub+"_rest3_corr.csv",index_col=0)
    df4 = pd.read_csv("rest/"+sub+"_rest4_corr.csv",index_col=0)
    hm1 = sns.heatmap(df1, cmap = 'jet')
    hm1.set(xlabel='Vertex', ylabel='Latent', title = sub+" Rest1 Correlation Matrix")
    plt.savefig("rest_plots/"+sub+"_rest1_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm2 = sns.heatmap(df2, cmap = 'jet')
    hm2.set(xlabel='Vertex', ylabel='Latent', title = sub+" Rest2 Correlation Matrix")
    plt.savefig("rest_plots/"+sub+"_rest2_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm3 = sns.heatmap(df3, cmap = 'jet')
    hm3.set(xlabel='Vertex', ylabel='Latent', title = sub+" Rest3 Correlation Matrix")
    plt.savefig("rest_plots/"+sub+"_rest3_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm4 = sns.heatmap(df4, cmap = 'jet')
    hm4.set(xlabel='Vertex', ylabel='Latent', title = sub+" Rest4 Correlation Matrix")
    plt.savefig("rest_plots/"+sub+"_rest4_corr.jpg", bbox_inches='tight')
    plt.clf()

# task
for sub in subs:
    df1 = pd.read_csv("task/"+sub+"_retbar1_corr.csv",index_col=0)
    df2 = pd.read_csv("task/"+sub+"_retbar2_corr.csv",index_col=0)
    df3 = pd.read_csv("task/"+sub+"_retccw_corr.csv",index_col=0)
    df4 = pd.read_csv("task/"+sub+"_retcon_corr.csv",index_col=0)
    df5 = pd.read_csv("task/"+sub+"_retcw_corr.csv",index_col=0)
    df6 = pd.read_csv("task/"+sub+"_retexp_corr.csv",index_col=0)

    hm1 = sns.heatmap(df1, cmap = 'jet')
    hm1.set(xlabel='Vertex', ylabel='Latent', title = sub+" RETBAR1 Correlation Matrix")
    plt.savefig("task_plots/"+sub+"_retbar1_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm2 = sns.heatmap(df2, cmap = 'jet')
    hm2.set(xlabel='Vertex', ylabel='Latent', title = sub+" RETBAR2 Correlation Matrix")
    plt.savefig("task_plots/"+sub+"_retbar2_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm3 = sns.heatmap(df3, cmap = 'jet')
    hm3.set(xlabel='Vertex', ylabel='Latent', title = sub+" RETCCW Correlation Matrix")
    plt.savefig("task_plots/"+sub+"_retccw_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm4 = sns.heatmap(df4, cmap = 'jet')
    hm4.set(xlabel='Vertex', ylabel='Latent', title = sub+" RETCON Correlation Matrix")
    plt.savefig("task_plots/"+sub+"_retcon_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm5 = sns.heatmap(df5, cmap = 'jet')
    hm5.set(xlabel='Vertex', ylabel='Latent', title = sub+" RETCW Correlation Matrix")
    plt.savefig("task_plots/"+sub+"_retcw_corr.jpg", bbox_inches='tight')
    plt.clf()
    hm6 = sns.heatmap(df6, cmap = 'jet')
    hm6.set(xlabel='Vertex', ylabel='Latent', title = sub+" RETEXP Correlation Matrix")
    plt.savefig("task_plots/"+sub+"_retexp_corr.jpg", bbox_inches='tight')
    plt.clf()
