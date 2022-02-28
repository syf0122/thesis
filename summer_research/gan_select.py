import pandas as pd
import numpy as np

ae_dir = '/data_qnap/yifeis/NAS/gan/with_param/losses/autoencoder_loss_'
di_dir = '/data_qnap/yifeis/NAS/gan/with_param/losses/discriminator_loss_'

l = 0
ae_loss_360_all_models = []
ae_loss_379_all_models = []
di_loss_360_all_models = []
di_loss_379_all_models = []

ae_360_min_lambda = -1
ae_379_min_lambda = -1
di_360_min_lambda = -1
di_379_min_lambda = -1

ae_360_min_value = 100
ae_379_min_value = 100
di_360_min_value = 100
di_379_min_value = 100


for n in range(99):
    l += 0.01
    l = round(l, 2)
    ae_loss_360 = pd.read_csv(ae_dir + str(l) + "_360.csv")
    ae_loss_379 = pd.read_csv(ae_dir + str(l) + "_379.csv")
    di_loss_360 = pd.read_csv(di_dir + str(l) + "_360.csv")
    di_loss_379 = pd.read_csv(di_dir + str(l) + "_379.csv")

    # get the mean
    # record the mean MSE
    ae_loss_360_mean = (ae_loss_360["0"] / (1-l)).mean()
    ae_loss_379_mean = (ae_loss_379["0"] / (1-l)).mean()
    di_loss_360_mean = (di_loss_360["0"] / l).mean()
    di_loss_379_mean = (di_loss_379["0"] / l).mean()
    ae_loss_360_all_models.append(ae_loss_360_mean)
    ae_loss_379_all_models.append(ae_loss_379_mean)
    di_loss_360_all_models.append(di_loss_360_mean)
    di_loss_379_all_models.append(di_loss_379_mean)

    # check and update the min mean MSE and lambda
    if ae_loss_360_mean < ae_360_min_value:
        ae_360_min_value = ae_loss_360_mean
        ae_360_min_lambda = l
    if ae_loss_379_mean < ae_379_min_value:
        ae_379_min_value = ae_loss_379_mean
        ae_379_min_lambda = l
    if di_loss_360_mean < di_360_min_value:
        di_360_min_value = di_loss_360_mean
        di_360_min_lambda = l
    if di_loss_379_mean < di_379_min_value:
        di_379_min_value = di_loss_379_mean
        di_379_min_lambda = l

print("The lowest mean MSE for autoencoder with epochs = 20, regions = 360, and lambda = {} has the MSE = {}".format(ae_360_min_lambda, ae_360_min_value))
print("The lowest mean MSE for autoencoder with epochs = 20, regions = 379, and lambda = {} has the MSE = {}".format(ae_379_min_lambda, ae_379_min_value))
print("The lowest mean MSE for discriminator with epochs = 20, regions = 360, and lambda = {} has the MSE = {}".format(di_360_min_lambda, di_360_min_value))
print("The lowest mean MSE for discriminator with epochs = 20, regions = 379, and lambda = {} has the MSE = {}".format(di_379_min_lambda, di_379_min_value))

# save the results
ae_resutls_360_df = pd.DataFrame()
ae_resutls_379_df = pd.DataFrame()
di_resutls_360_df = pd.DataFrame()
di_resutls_379_df = pd.DataFrame()
ae_resutls_360_df['lambda'] = list(np.arange(0.0, 1.0, 0.01))[1:]
ae_resutls_379_df['lambda'] = list(np.arange(0.0, 1.0, 0.01))[1:]
di_resutls_360_df['lambda'] = list(np.arange(0.0, 1.0, 0.01))[1:]
di_resutls_379_df['lambda'] = list(np.arange(0.0, 1.0, 0.01))[1:]
ae_resutls_360_df['avg MSE'] = ae_loss_360_all_models
ae_resutls_379_df['avg MSE'] = ae_loss_379_all_models
di_resutls_360_df['avg MSE'] = di_loss_360_all_models
di_resutls_379_df['avg MSE'] = di_loss_379_all_models

ae_resutls_360_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/Averaged_MSE_for_ae_360_Models.csv')
ae_resutls_379_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/Averaged_MSE_for_ae_379_Models.csv')
di_resutls_360_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/Averaged_MSE_for_disc_360_Models.csv')
di_resutls_379_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/Averaged_MSE_for_disc_379_Models.csv')
