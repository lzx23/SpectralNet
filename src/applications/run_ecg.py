# get all the imports from spectralnet

import sys, os, pickle
import tensorflow as tf
import numpy as np
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop

# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from core import train
from core import costs
from core import networks
from core.layer import stack_layers
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds

import wfdb
from collections import defaultdict
import matplotlib.pyplot as plt

from run_net_no_clustering import run_net_no_clustering

# import data
def get_ecg_data(n = 0, id = '16265', folder = '/Users/lianexu/Desktop/UROP/mit-bih-normal-sinus-rhythm-database-1.0.0', shuffle = False):
    """
    Input:
    n - 0 or 1
    id

    Output:
    data, in format for run_net_no_clustering
    """
    assert (n == 0 or n == 1)

    # import data
    # record = wfdb.rdrecord(folder + os.sep + id, pb_dir='nsrdb/')
    record = wfdb.rdrecord(folder + os.sep + id)
    # wfdb.plot_wfdb(record=record, title='Record '+ id +' from Physionet NSRDB') 
    # print(record.__dict__)

    ecg = record.__dict__['p_signal'][:, n]

    window = 12
    overlap = 6
    X = lag_map(ecg, window = window, overlap = overlap)
    num_points = np.shape(X)[0]
    y = np.empty(num_points)

    # split into train, val sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .25, shuffle = shuffle)

    # make data dictionary
    data = {}
    data['spectral'] = {}
    data['spectral']['train_and_test'] = X_train, y_train, X_val, y_val, None, None
    data['spectral']['train_unlabeled_and_labeled'] = X_train, y_train, np.empty((0, window)), np.empty(0)
    data['spectral']['val_unlabeled_and_labeled'] = X_val, y_val, None, None

    return data, num_points


def lag_map(signal, window = 12, overlap = 6):

    step = window - overlap
    padded_signal = np.hstack((np.zeros(window//2), signal))
    padded_signal = np.hstack((padded_signal, np.zeros(window//2)))

    return np.array([padded_signal[i:i+window] 
    for i in np.arange(0, len(padded_signal)-window, step = window-overlap)])

def plot(data, x_spectralnet):
    """
    plots the values along two eigenvectors
    """

    num_eig = np.shape(x_spectralnet)[1]
    fig, axs = plt.subplots(num_eig * (num_eig - 1)/2)

    x_train, _, x_val, _, _, _ = data['spectral']['train_and_test']
    x = np.concatenate((x_train, x_val), axis=0)
    x_means = np.mean(x, axis = 1)

    count = 0

    for i in range(num_eig):
        eig1 = x_spectralnet[:, i]
        for j in range(i+1, num_eig):
            eig2 = x_spectralnet[:, j]
            axs[count].scatter(eig1, eig2, c = x_means, cmap = 'plasma')
            axs[count].set_xlabel('Eigenvector %g' % i)
            axs[count].set_ylabel('Eigenvector %g' % j)
            count += 1
    
    plt.show()


params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        # 'dset': args.dset,                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        }
params.update(general_params)

data, num_points = get_ecg_data()

print("Number of points: %g" % num_points)

# ecg_params = {
#     # training parameters
#     'n_clusters': 3,
#     'use_code_space': False,
#     'affinity': 'full',
#     'n_nbrs': 10,
#     'scale_nbr': 2,
#     'spec_ne': 100,
#     'spec_lr': 1e-3,
#     'spec_patience': 30,
#     'spec_drop': 0.1,
#     'batch_size': 128,
#     'batch_size_orthonorm': 128,
#     'spec_reg': None,
#     'arch': [
#         {'type': 'softplus', 'size': 50},
#         {'type': 'BatchNormalization'},
#         {'type': 'softplus', 'size': 50},
#         {'type': 'BatchNormalization'},
#         {'type': 'softplus', 'size': 50},
#         {'type': 'BatchNormalization'},
#         ],
#     'use_all_data': True,
#     }
# ecg_params = {
#         'n_clusters': 3,
#         'affinity': 'siamese',
#         'n_nbrs': 30,
#         'scale_nbr': 10,
#         'siam_k': 100,
#         'siam_ne': 20,
#         'spec_ne': 300,
#         'siam_lr': 1e-3,
#         'spec_lr': 5e-5,
#         'siam_patience': 1,
#         'spec_patience': 5,
#         'siam_drop': 0.1,
#         'spec_drop': 0.1,
#         'batch_size': 2048,
#         'siam_reg': 1e-2,
#         'spec_reg': 5e-1,
#         'siam_n': None,
#         'siamese_tot_pairs': 400000,
#         'arch': [
#             {'type': 'relu', 'size': 512},
#             {'type': 'relu', 'size': 256},
#             {'type': 'relu', 'size': 128},
#             ],
#         'use_approx': True,
#         'use_all_data': True,
#         }

ecg_params = {
        'n_clusters': 4,
        'affinity': 'full',
        'n_nbrs': 30,
        'scale_nbr': 10,
        'spec_ne': 300,
        'spec_lr': 5e-5,
        'spec_patience': 5,
        'spec_drop': 0.1,
        'batch_size': 2048,
        'spec_reg': 5e-1,
        'arch': [
            {'type': 'relu', 'size': 512},
            {'type': 'relu', 'size': 256},
            {'type': 'relu', 'size': 128},
            ],
        }

params.update(ecg_params)

x_spectralnet = run_net_no_clustering(data, params)
plot(data, x_spectralnet)