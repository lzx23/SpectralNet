#%%
from run_net_no_clustering import run_net_no_clustering, make_data
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D

Axes3D

# ----------------------------------------------------------------------
# Locally linear embedding of the swiss roll

from sklearn import manifold, datasets

n = 1500
X, color = datasets.make_swiss_roll(n_samples=n)

# data = make_data(X, shuffle = False)

#%%
    # Function to plot result
def plot_swissroll(X, color, X_r, title = "", eig1 = 0, eig2 = 1):
    '''
    Plots swiss roll and projected data

    X : swiss roll data
    color : colors for plot
    X_r : projected data
    '''
    fig = plt.figure()

    ax = fig.add_subplot(211, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

    ax.set_title("Original data")
    ax = fig.add_subplot(212)
    ax.scatter(X_r[:, eig1], X_r[:, eig2], c=color, cmap=plt.cm.Spectral)
    plt.xlabel("Eigenvector %g" % eig1)
    plt.ylabel("Eigenvector %g" % eig2)
    # plt.axis("tight")
    plt.subplots_adjust(bottom=0.1,  
                        top=1, 
                        wspace=0.4, 
                        hspace=0.4)
    plt.xticks([]), plt.yticks([])
    plt.title("Projected data " + title)
    plt.show()

#%%
# print("Computing LLE embedding")
# X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
# print("Done. Reconstruction error: %g" % err)

# SpectralNet for swiss roll
siamese = True                          # use siamese net or not
num_eig = 3                                   # number of eigenvectors (also number of clusters)
# def run_spectralnet_swissroll(X = None, color = None, siamese = True, n = 3):
#     '''
#     runs spectralnet on a swiss roll

#     X : swiss roll data. If None, generates swiss roll data
#     siamese: use siamese net or not
#     n: number of eigenvectors (also number of clusters)
#     '''

if X is None:
    X, color = datasets.make_swiss_roll(n_samples=n)

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        # 'dset': args.dset,                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        }
params.update(general_params)

if siamese:
    swissroll_params = {
        # training parameters
        'n_clusters': num_eig,
        'use_code_space': False,
        'affinity': 'siamese',
        'n_nbrs': 10,
        'scale_nbr': 2,
        'siam_k': 2,                        # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
                                                # a 'positive' pair by siamese net

        'siam_ne': 400,                     # number of training epochs for siamese net
        'spec_ne': 300,
        'siam_lr': 1e-3,                    # initial learning rate for siamese net
        'spec_lr': 1e-3,
        'siam_patience': 10,                # early stopping patience for siamese net
        'spec_patience': 30,
        'siam_drop': 0.1,                   # learning rate scheduler decay for siamese net
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 128,
        'siam_reg': None,                   # regularization parameter for siamese net
        'siam_n': None,                     # subset of the dataset used to construct training pairs for siamese net
        'spec_reg': None,
        'siamese_tot_pairs': 600000,        # total number of pairs for siamese net
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'use_all_data': True,
        }
    params.update(swissroll_params)
    print("Formatting data")
    data = make_data(X, shuffle = False, params = params)
else:
    swissroll_params = {
        # training parameters
        'n_clusters': 2,
        'use_code_space': False,
        'affinity': 'full',
        'n_nbrs': 10,
        'scale_nbr': 2,
        'spec_ne': 300,
        'spec_lr': 1e-3,
        'spec_patience': 30,
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 128,
        'spec_reg': None,
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'use_all_data': True,
        }
    params.update(swissroll_params)
    print("Formatting data")
    data = make_data(X, shuffle = False)

n_nbrs = [5, 10, 25, 100]
scale_nbrs = [2, 5, 10, 25, 100]

for nbrs in n_nbrs:

    params['n_nbrs'] = nbrs
    print("n_nbrs = " + str(nbrs))
    print("Running SpectralNet on swiss roll")
    X_r = run_net_no_clustering(data, params)

    print("Done. Now plotting.")
    title = 'n_nbrs = ' + str(params['n_nbrs']) + ', scale_nbr = ' + str(params['scale_nbr'])
    plot_swissroll(X, color, X_r, title = title)

params['n_nbrs'] = 10
scale_nbrs = [2, 5, 10, 25, 100]
for nbrs in scale_nbrs:

    params['scale_nbr'] = nbrs
    print("scale_nbr = " + str(nbrs))
    print("Running SpectralNet on swiss roll")
    X_r = run_net_no_clustering(data, params)

    print("Done. Now plotting.")
    title = 'n_nbrs = ' + str(params['n_nbrs']) + ', scale_nbr = ' + str(params['scale_nbr'])
    plot_swissroll(X, color, X_r, title = title)

params['n_nbrs'] = 10
params['scale_nbr'] = 2
for _ in range(5):
    print("Running SpectralNet on swiss roll")
    X_r = run_net_no_clustering(data, params)

    print("Done. Now plotting.")
    title = 'n_nbrs = ' + str(params['n_nbrs']) + ', scale_nbr = ' + str(params['scale_nbr'])
    plot_swissroll(X, color, X_r, title = title)

if __name__ == '__main__':
    # generate swiss roll
    n = 1500
    X, color = datasets.make_swiss_roll(n_samples=n)