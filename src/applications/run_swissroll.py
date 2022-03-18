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

data = make_data(X, shuffle = False)
#%%
# print("Computing LLE embedding")
# X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
# print("Done. Reconstruction error: %g" % err)

# SpectralNet for swiss roll

print("Running SpectralNet on swiss roll")

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        # 'dset': args.dset,                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        }
params.update(general_params)

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

X_r = run_net_no_clustering(data, params)

print("Done")
#%%%
# ----------------------------------------------------------------------
# Plot result

fig = plt.figure()

ax = fig.add_subplot(211, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")
ax = fig.add_subplot(212)
eig1 = 0
eig2 = 1
ax.scatter(X_r[:, eig1], X_r[:, eig2], c=color, cmap=plt.cm.Spectral)
plt.xlabel("Eigenvector %g" % eig1)
plt.ylabel("Eigenvector %g" % eig2)
plt.axis("tight")
plt.xticks([]), plt.yticks([])
plt.title("Projected data")
plt.show()
# %%
