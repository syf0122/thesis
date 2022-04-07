from sphericalunet.utils.utils import *
from sphericalunet.utils.interp_numpy  import *
from sphericalunet.utils.vtk import read_vtk, write_vtk
import numpy as np
import itertools
from sklearn.neighbors import KDTree
import math, multiprocessing, os


template_163842 = read_vtk("/data_qnap/yifeis/spherical_cnn/example_data/test1.lh.160k.vtk")
neigh_orders_163842 = get_neighs_order('adj_mat_order_163842_rotated_0.mat')

file = "/data_qnap/yifeis/HCP_7T/100610/rest1_left.vtk"

data = read_vtk(file)
print(np.max(data['cdata']))
print(np.min(data['cdata']))
print(data['cdata'][:, np.newaxis].shape)

def singleVertexInterpo_tseries(vertex, vertices, tree, neigh_orders, feat):
    """
    Compute the three indices and weights for sphere interpolation at given position.
    """
    # print(feat.shape)# 163482, 1, 900
    # print(vertex.shape) # 3
    # print(vertex[np.newaxis,:])
    _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
    top3_near_vertex_index = np.squeeze(top3_near_vertex_index)
    if isATriangle(neigh_orders, top3_near_vertex_index):
        v0 = vertices[top3_near_vertex_index[0]]
        v1 = vertices[top3_near_vertex_index[1]]
        v2 = vertices[top3_near_vertex_index[2]]
        normal = np.cross(v1-v2, v0-v2)

        vertex_proj = v0.dot(normal)/vertex.dot(normal) * vertex

        area_BCP = np.linalg.norm(np.cross(v2-vertex_proj, v1-vertex_proj))/2.0
        area_ACP = np.linalg.norm(np.cross(v2-vertex_proj, v0-vertex_proj))/2.0
        area_ABP = np.linalg.norm(np.cross(v1-vertex_proj, v0-vertex_proj))/2.0
        area_ABC = np.linalg.norm(normal)/2.0

        if area_BCP + area_ACP + area_ABP - area_ABC > 1e-5:
             inter_indices, inter_weight = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders)
        else:
            inter_weight = np.array([area_BCP, area_ACP, area_ABP])
            inter_weight = inter_weight / inter_weight.sum()
            inter_indices = top3_near_vertex_index
    else:
        inter_indices, inter_weight = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders)

    # print(inter_weight.shape) # 3 (1, 0, 0)
    # print(inter_weight[:,np.newaxis].shape) # 3, 1
    # print(feat.shape[1]) # 1
    # print(inter_indices) # [a, b, c]
    # print(feat[inter_indices].shape) # 3, 1, 900
    # print(inter_indices)
    w = inter_weight[:,np.newaxis]
    w = w[:, np.newaxis] # 3,1,1
    w = np.repeat(w, feat.shape[2], axis=2) # 3, 1, # of timepoints
    return np.sum(np.multiply(feat[inter_indices, :, :], w), axis=0)

def multiVertexInterpo_tseries(vertexs, vertices, tree, neigh_orders, feat):
    feat_inter = np.zeros((vertexs.shape[0], feat.shape[1], feat.shape[2]))
    for i in range(vertexs.shape[0]):
        single = singleVertexInterpo_tseries(vertexs[i,:], vertices, tree, neigh_orders, feat)
        feat_inter[i,:,:] = single
    return feat_inter

def resampleSphereSurf_tseries(vertices_fix, vertices_inter, feat, faces=None, upsample_neighbors=None, neigh_orders=None):
    """
    resample sphere surface
    Parameters
    ----------
    vertices_fix :  N*3, numpy array
        DESCRIPTION.
    vertices_inter : unknown*3, numpy array, points to be interpolated
        DESCRIPTION.
    feat :  N*D, features to be interpolated
        DESCRIPTION.
    faces :  N*4, numpy array, the first column shoud be all 3
        is the original faces directly read using read_vtk,. The default is None.
    upsample_neighbors : TYPE, optional
        DESCRIPTION. The default is None.
    neigh_orders : TYPE, optional
        DESCRIPTION. The default is None.
    Returns
    -------
    """
    assert vertices_fix.shape[0] == feat.shape[0], "vertices.shape[0] == feat.shape[0], error"
    assert vertices_fix.shape[1] == 3, "vertices size not right"

    vertices_fix = vertices_fix.astype(np.float64)
    vertices_inter = vertices_inter.astype(np.float64)
    feat = feat.astype(np.float64)

    vertices_fix = vertices_fix / np.linalg.norm(vertices_fix, axis=1)[:,np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:,np.newaxis]  # normalize to 1

    if len(feat.shape) == 1:
        feat = feat[:,np.newaxis]

    if neigh_orders is None:
        if faces is not None:
            assert faces.shape[1] == 4, "faces shape is wrong, should be N*4"
            assert (faces[:,0] == 3).sum() == faces.shape[0], "the first column of faces should be all 3"
            faces = faces[:,[1,2, 3]]
            neigh_orders = np.zeros((vertices_fix.shape[0],30), dtype=np.int64)-1
            for i in range(faces.shape[0]):
                if faces[i,1] not in neigh_orders[faces[i,0]]:
                    neigh_orders[faces[i,0], np.where(neigh_orders[faces[i,0]] == -1)[0][0]] = faces[i,1]
                if faces[i,2] not in neigh_orders[faces[i,0]]:
                    neigh_orders[faces[i,0], np.where(neigh_orders[faces[i,0]] == -1)[0][0]] = faces[i,2]
                if faces[i,0] not in neigh_orders[faces[i,1]]:
                    neigh_orders[faces[i,1], np.where(neigh_orders[faces[i,1]] == -1)[0][0]] = faces[i,0]
                if faces[i,2] not in neigh_orders[faces[i,1]]:
                    neigh_orders[faces[i,1], np.where(neigh_orders[faces[i,1]] == -1)[0][0]] = faces[i,2]
                if faces[i,1] not in neigh_orders[faces[i,2]]:
                    neigh_orders[faces[i,2], np.where(neigh_orders[faces[i,2]] == -1)[0][0]] = faces[i,1]
                if faces[i,0] not in neigh_orders[faces[i,2]]:
                    neigh_orders[faces[i,2], np.where(neigh_orders[faces[i,2]] == -1)[0][0]] = faces[i,0]

        else:
            neigh_orders = get_neighs_order(abspath+'/neigh_indices/adj_mat_order_'+ str(vertices_fix.shape[0]) +'.mat')
            neigh_orders = neigh_orders.reshape(vertices_fix.shape[0], 7)
    else:
        neigh_orders = neigh_orders.reshape(vertices_fix.shape[0], 7)

    feat_inter = np.zeros((vertices_inter.shape[0], feat.shape[1], feat.shape[2]))
    tree = KDTree(vertices_fix, leaf_size=10)  # build kdtree

    """ multiple processes method: 163842: 9.6s, 40962: 2.8s, 10242: 1.0s, 2562: 0.28s """
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    vertexs_num_per_cpu = math.ceil(vertices_inter.shape[0]/cpus)
    results = []

    for i in range(cpus):
        results.append(pool.apply_async(multiVertexInterpo_tseries, args=(vertices_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:], vertices_fix, tree, neigh_orders, feat,)))

    pool.close()
    pool.join()

    for i in range(cpus):
        result_i = results[i].get()
        feat_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:,:] = result_i
    return np.squeeze(feat_inter)



resampled_feat = resampleSphereSurf_tseries(data['vertices'],
                                    template_163842['vertices'],
                                    data['cdata'][:, np.newaxis],
                                    neigh_orders=neigh_orders_163842)
print(resampled_feat.shape)
print(np.max(resampled_feat))
print(np.min(resampled_feat))
surf = {'vertices': template_163842['vertices'],
        'faces': template_163842['faces'],
        'cdata': resampled_feat}
print("All zeros: {}".format(np.all(resampled_feat == 0)))
# write_vtk(surf, file.replace('.vtk', '.resample.vtk'))
