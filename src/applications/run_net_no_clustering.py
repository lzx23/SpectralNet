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
from core import pairs

import wfdb
from collections import defaultdict
import matplotlib.pyplot as plt

def make_data(X, y = None, shuffle = True, params = None):
    """
    given X, y, splits into training and validation set and returns it in the format for run_net_no_clustering

    Parameters:
    X : np array
        each row is a datapoint

    y: 1d np array, optional
        y values corresponding to X

    """
    if y is None:
        y = np.empty(np.shape(X)[0])

    # split into train, val sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .25, shuffle = shuffle)
    X_train_labeled = np.empty((0, np.shape(X_train)[1]))
    y_train_labeled = np.empty(0)
    # X_val_labeled = np.empty((0, np.shape(X_train)[1]))
    # y_val_labeled = np.empty(0)

    # make data dictionary
    ret = {}
    ret['spectral'] = {}
    ret['spectral']['train_and_test'] = X_train, y_train, X_val, y_val, None, None
    ret['spectral']['train_unlabeled_and_labeled'] = X_train, y_train, X_train_labeled, y_train_labeled
    ret['spectral']['val_unlabeled_and_labeled'] = X_val, y_val, None, None

    if params is not None and 'siamese' in params['affinity']:

        ret['siamese'] = {}

        if params.get('precomputedKNNPath'):
            # if we use precomputed knn, we cannot shuffle the data; instead
            # we pass the permuted index array and the full matrix so that
            # create_pairs_from_unlabeled data can keep track of the indices
            p_train_unlabeled = p_train[:len(x_train_unlabeled)]
            train_path = params.get('precomputedKNNPath', '')
            if train_val_split[1] < 0.09 or params['siam_k'] > 100:
                # if the validation set is very small, the benefit of
                # the precomputation is small, and there will be a high miss
                # rate in the precomputed neighbors (neighbors that are not
                # in the validation set) so we just recomputed neighbors
                p_val_unlabeled = None
                val_path = ''
            else:
                p_val_unlabeled = p_val[:len(x_val_unlabeled)]
                val_path = params.get('precomputedKNNPath', '')
        else:
            # if we do not use precomputed knn, then this does not matter
            p_train_unlabeled = None
            train_path = params.get('precomputedKNNPath', '')
            p_val_unlabeled = None
            val_path = params.get('precomputedKNNPath', '')

        pairs_train_unlabeled, dist_train_unlabeled = pairs.create_pairs_from_unlabeled_data(
            x1=X_train,
            p=p_train_unlabeled,
            k=params.get('siam_k'),
            tot_pairs=params.get('siamese_tot_pairs'),
            precomputed_knn_path=train_path,
            use_approx=params.get('use_approx', False),
            pre_shuffled=True,
        )
        pairs_val_unlabeled, dist_val_unlabeled = pairs.create_pairs_from_unlabeled_data(
            x1=X_val,
            p=p_val_unlabeled,
            k=params.get('siam_k'),
            tot_pairs=params.get('siamese_tot_pairs'),
            precomputed_knn_path=val_path,
            use_approx=params.get('use_approx', False),
            pre_shuffled=True,
        )

        # #get pairs for labeled data
        # class_indices = [np.where(y_train_labeled == i)[0] for i in range(params['n_clusters'])]
        # pairs_train_labeled, dist_train_labeled = pairs.create_pairs_from_labeled_data(x_train_labeled, class_indices)
        # class_indices = [np.where(y_val_labeled == i)[0] for i in range(params['n_clusters'])]
        # pairs_val_labeled, dist_val_labeled = pairs.create_pairs_from_labeled_data(x_val_labeled, class_indices)

        # ret['siamese']['train_unlabeled_and_labeled'] = (pairs_train_unlabeled, dist_train_unlabeled, pairs_train_labeled, dist_train_labeled)
        # ret['siamese']['val_unlabeled_and_labeled'] = (pairs_val_unlabeled, dist_val_unlabeled, pairs_val_labeled, dist_val_labeled)

        ret['siamese']['train_unlabeled_and_labeled'] = (pairs_train_unlabeled, dist_train_unlabeled, np.empty((0, np.shape(pairs_train_unlabeled)[1])), np.empty(0))
        ret['siamese']['val_unlabeled_and_labeled'] = (pairs_val_unlabeled, dist_val_unlabeled, np.empty((0, np.shape(pairs_val_unlabeled)[1])), np.empty(0))


        #combine labeled and unlabeled pairs for training the siamese
        # pairs_train = np.concatenate((pairs_train_unlabeled, pairs_train_labeled), axis=0)
        # dist_train = np.concatenate((dist_train_unlabeled, dist_train_labeled), axis=0)
        # pairs_val = np.concatenate((pairs_val_unlabeled, pairs_val_labeled), axis=0)
        # dist_val = np.concatenate((dist_val_unlabeled, dist_val_labeled), axis=0)

        ret['siamese']['train_and_test'] = (pairs_train_unlabeled, dist_train_unlabeled, pairs_val_unlabeled, dist_val_unlabeled)        

    return ret

def run_net_no_clustering(data, params):

    #
    # UNPACK DATA
    #

    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
    x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral']['train_unlabeled_and_labeled']
    x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

    if 'siamese' in params['affinity']:
        pairs_train, dist_train, pairs_val, dist_val = data['siamese']['train_and_test']

    x = np.concatenate((x_train, x_val), axis=0)
    # y = np.concatenate((y_train, y_val, y_test), axis=0)

    # y_train_labeled_onehot = np.empty((0, len(np.unique(y))))
    y_train_labeled_onehot = None

    #
    # SET UP INPUTS
    #

    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')
    # y_true = tf.placeholder(tf.float32, shape=(None, 0), name='y_true')

    batch_sizes = {
            'Unlabeled': params['batch_size'],
            'Labeled': params['batch_size'],
            'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
            }

    input_shape = x.shape[1:]

    # spectralnet has three inputs -- they are defined here
    inputs = {
            'Unlabeled': Input(shape=input_shape,name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape,name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape,name='OrthonormInput'),
            }

    #
    # DEFINE AND TRAIN SIAMESE NET
    #

    # run only if we are using a siamese network
    if params['affinity'] == 'siamese':
        siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), y_true)

        history = siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
                params['siam_lr'], params['siam_drop'], params['siam_patience'],
                params['siam_ne'], params['siam_batch_size'])

    else:
        siamese_net = None

    #
    # DEFINE AND TRAIN SPECTRALNET
    #
    # class SpectralNet:
    # def __init__(self, inputs, arch, spec_reg, y_true, y_train_labeled_onehot,
    #         n_clusters, affinity, scale_nbr, n_nbrs, batch_sizes,
    #         siamese_net=None, x_train=None, have_labeled=False)

    # spectral_net = networks.SpectralNet(inputs, params['arch'],
    #         params.get('spec_reg'), y_true, y_train_labeled_onehot,
    #         params['n_clusters'], params['affinity'], params['scale_nbr'],
    #         params['n_nbrs'], batch_sizes, siamese_net)

    spectral_net = networks.SpectralNet(inputs, params['arch'],
        params.get('spec_reg'), y_true, y_train_labeled_onehot,
        params['n_clusters'], params['affinity'], params['scale_nbr'],
        params['n_nbrs'], batch_sizes, siamese_net, x_train, len(x_train_labeled))

    # def train(self, x_train_unlabeled, x_train_labeled, x_val_unlabeled,
    #         lr, drop, patience, num_epochs):

    x_train_labeled = None

    print("start training")

    spectral_net.train(
            x_train_unlabeled, x_train_labeled, x_val_unlabeled,
            params['spec_lr'], params['spec_drop'], params['spec_patience'],
            params['spec_ne'])

    print("finished training")

    #
    # EVALUATE
    #

    # get final embeddings
    return spectral_net.predict(x)