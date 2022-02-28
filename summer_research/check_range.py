import os
import numpy as np

def load_OAS_data(gp, subjects, resolution, prepro):
    dir = '/data_qnap/yifeis/NAS/data/'+gp+'_p/'
    processed_files = os.listdir(dir)
    processed_files.sort()
    sessions = {}
    data = {}
    for subject in subjects:
        sub_data = []
        sub_session = []
        for f in processed_files:
            if subject in f and ("_"+resolution+"_"+prepro+"_p.npy") in f:
                mtx_p = np.load(dir + f) # (149, 360)
                # # normalize in the spatial domain
                # mtx_p = mtx_p.T
                # mtx_p = (mtx_p - np.mean(mtx_p,axis=0))/np.std(mtx_p,axis=0)
                # mtx_p = mtx_p.T
                sub_data.append(mtx_p)
                sub_session.append(f[13:18])
                print(np.max(mtx_p))
                print(np.min(mtx_p))
                print()
        data[subject] = sub_data
        sessions[subject] = sub_session
    return (data, sessions)

def load_HCP_data(subjects):
    dir = '/data_qnap/yifeis/new/processed/'
    sessions = {}
    data = {}
    data_360 = {}
    for subject in subjects:
        sub_data = []
        sub_data_360 = []
        sub_session = []
        processed_files = os.listdir(dir + subject + "/")
        processed_files.sort()
        for f in processed_files:
            if "rest" in f and "_p.npy" in f:
                mtx_p = np.load(dir + subject + "/" + f) # (900, 379)
                mtx_p_360 = mtx_p[:, :360]
                # # normalize in the spatial domain
                # mtx_p = mtx_p.T
                # mtx_p = (mtx_p - np.mean(mtx_p,axis=0))/np.std(mtx_p,axis=0)
                # mtx_p = mtx_p.T
                # mtx_p_360 = mtx_p_360.T
                # mtx_p_360 = (mtx_p_360 - np.mean(mtx_p_360,axis=0))/np.std(mtx_p_360,axis=0)
                # mtx_p_360 = mtx_p_360.T
                print(np.max(mtx_p))
                print(np.min(mtx_p))
                print()
                print(np.max(mtx_p_360))
                print(np.min(mtx_p_360))
                print()
                sub_data.append(mtx_p)
                sub_data_360.append(mtx_p_360)
                sub_session.append(f[:5])
        data[subject] = sub_data
        data_360[subject] = sub_data_360
        sessions[subject] = sub_session
    return (data, data_360, sessions)

test_data_360_2mm, test_sessions_360_2mm = load_OAS_data('HC', ['sub-OAS30001'], "2mm", 'norm')
test_data_360_4mm, test_sessions_360_4mm = load_OAS_data('HC', ['sub-OAS30001'], "4mm", 'norm')
test_data_379, test_sessions_379 = load_OAS_data('HC', ['sub-OAS30001'], 'surf', 'norm')

hcp_data_379, hcp_data_360, hcp_session = load_HCP_data(['100610'])
