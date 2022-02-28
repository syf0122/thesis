import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

hc_quality = pd.read_csv('/data_qnap/yifeis/NAS/data/quality/HC_mean_param.csv', index_col=0)
ad_quality = pd.read_csv('/data_qnap/yifeis/NAS/data/quality/AD_mean_param.csv', index_col=0)

print(hc_quality.shape)
print(ad_quality.shape)

# distribution
sns.distplot(hc_quality.std_dvars).set_title("OASIS HC Standardized DVARS")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/HC_dvars_dist.png')
plt.clf()
sns.distplot(hc_quality.framewise_displacement).set_title("OASIS HC FD")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/HC_fd_dist.png')
plt.clf()
sns.distplot(ad_quality.std_dvars).set_title("OASIS AD Standardized DVARS")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/AD_dvars_dist.png')
plt.clf()
sns.distplot(ad_quality.framewise_displacement).set_title("OASIS AD FD")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/AD_fd_dist.png')
plt.clf()

# boxplot
sns.boxplot(hc_quality.std_dvars).set_title("OASIS HC Standardized DVARS")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/HC_dvars_box.png')
plt.clf()
sns.boxplot(hc_quality.framewise_displacement).set_title("OASIS HC FD")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/HC_fd_box.png')
plt.clf()
sns.boxplot(ad_quality.std_dvars).set_title("OASIS AD Standardized DVARS")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/AD_dvars_box.png')
plt.clf()
sns.boxplot(ad_quality.framewise_displacement).set_title("OASIS AD FD")
plt.savefig('/data_qnap/yifeis/NAS/data/quality/AD_fd_box.png')
