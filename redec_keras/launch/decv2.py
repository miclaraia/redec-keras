import pickle
import numpy as np
import os
import shutil
from datetime import datetime

from redec_keras.models.decv2 import DECv2, Config


def train(
        splits_file,
        save_dir, 
        model_name,
        batch_size,
        lr,
        momentum,
        tol,
        maxiter,
        n_clusters,
        save_interval,
        name):
    if os.path.isdir(save_dir):
        raise FileExistsError(save_dir)
    os.makedirs(save_dir)

    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)

    x_test, y_test = splits['test']
    x_train, y_train = splits['train']
    x_valid, y_valid = splits['valid']
    x_train_dev, y_train_dev = splits['train_dev']

    order = np.random.permutation(x_train.shape[0])
    print(x_train.shape, y_train.shape)
    x_train = x_train[order,:]
    y_train = y_train[order]
    print(x_train.shape, x_test.shape, x_valid.shape, x_train_dev.shape)

    config_args = {
        'save_dir': save_dir,
        'name': name,
        'source_dir': None,
        'splits_file': splits_file,
        'n_classes': 2,
        'n_clusters': n_clusters,
        'batch_size': batch_size,
        'optimizer': ('SGD', {'lr': lr, 'momentum': momentum}),
        'tol': tol,
        'maxiter': maxiter,
        'save_interval': save_interval,
        'source_weights': (None, None),
    }

    ae_weights = os.path.join(save_dir, 'ae_weights.h5')
    dec_weights = os.path.join(save_dir, 'DEC_model_final.h5')
    config_args['save_weights'] = ae_weights, dec_weights

    config = Config(**config_args)
    config.dump()

    dec = DECv2(config, x_train.shape)
    dec.init(x_train)

    y_pred = dec.clustering(
        (x_train, y_train),
        (x_train_dev, y_train_dev),
        (x_valid, y_valid))

    dec.report_run(splits)