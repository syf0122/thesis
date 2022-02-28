import pandas as pd
import numpy as np

ae_dir = '/data_qnap/yifeis/NAS/gan/with_param/losses/autoencoder_loss_'
di_dir = '/data_qnap/yifeis/NAS/gan/with_param/losses/discriminator_loss_'

l = 0
ae_loss_360_2_all_models = []
ae_loss_360_4_all_models = []
ae_loss_379_all_models = []

ae_360_2_min_lambda = -1
ae_360_4_min_lambda = -1
ae_379_min_lambda   = -1

ae_360_2_min_value = 100
ae_360_4_min_value = 100
ae_379_min_value   = 100


for n in range(99):
    l += 0.01
    l = round(l, 2)
    ae_loss_360_2 = pd.read_csv(ae_dir + '2mm_' + str(l) + "_360.csv")
    ae_loss_360_4 = pd.read_csv(ae_dir + '4mm_' + str(l) + "_360.csv")
    ae_loss_379 = pd.read_csv(ae_dir + 'surf_' + str(l) + "_379.csv")

    # get the mean
    # record the mean MSE
    ae_loss_360_2_last = (ae_loss_360_2["0"] / (1-l)).iloc[-1]
    ae_loss_360_4_last = (ae_loss_360_4["0"] / (1-l)).iloc[-1]
    ae_loss_379_last = (ae_loss_379["0"] / (1-l)).iloc[-1]
    ae_loss_360_2_all_models.append(ae_loss_360_2_last)
    ae_loss_360_4_all_models.append(ae_loss_360_4_last)
    ae_loss_379_all_models.append(ae_loss_379_last)

    # check and update the min mean MSE and lambda
    if ae_loss_360_2_last < ae_360_2_min_value:
        ae_360_2_min_value = ae_loss_360_2_last
        ae_360_2_min_lambda = l
    if ae_loss_360_4_last < ae_360_4_min_value:
        ae_360_4_min_value = ae_loss_360_4_last
        ae_360_4_min_lambda = l
    if ae_loss_379_last < ae_379_min_value:
        ae_379_min_value = ae_loss_379_last
        ae_379_min_lambda = l


print("The minimum last MSE for 2mm autoencoder with epochs = 10, regions = 360, and lambda = {} has the MSE = {}".format(ae_360_2_min_lambda, ae_360_2_min_value))
print("The minimum last MSE for 4mm autoencoder with epochs = 10, regions = 360, and lambda = {} has the MSE = {}".format(ae_360_4_min_lambda, ae_360_4_min_value))
print("The minimum last MSE for autoencoder with epochs = 10, regions = 379, and lambda = {} has the MSE = {}".format(ae_379_min_lambda, ae_379_min_value))

# # save the results
# ae_resutls_360_2_df = pd.DataFrame()
# ae_resutls_360_4_df = pd.DataFrame()
# ae_resutls_379_df = pd.DataFrame()
# ae_resutls_360_2_df['lambda'] = list(np.arange(0.0, 1.0, 0.01))[1:]
# ae_resutls_360_4_df['lambda'] = list(np.arange(0.0, 1.0, 0.01))[1:]
# ae_resutls_379_df['lambda'] = list(np.arange(0.0, 1.0, 0.01))[1:]
# ae_resutls_360_2_df['avg MSE'] = ae_loss_360_2_all_models
# ae_resutls_360_4_df['avg MSE'] = ae_loss_360_4_all_models
# ae_resutls_379_df['avg MSE'] = ae_loss_379_all_models
#
# ae_resutls_360_2_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/Averaged_MSE_for_ae_360_Models_2mm.csv')
# ae_resutls_360_4_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/Averaged_MSE_for_ae_360_Models_4mm.csv')
# ae_resutls_379_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/Averaged_MSE_for_ae_379_Models.csv')
