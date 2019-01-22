import pickle
import numpy as np
import os
import shutil
from datetime import datetime

from redec_keras.models.multitask import MultitaskDEC, Config
from redec_keras.models.decv2 import Config as SourceConfig


def train(
        source_dir,
        save_dir,
        model_name,
        batch_size,
        lr,
        momentum,
        tol,
        maxiter,
        name,
        save_interval,
        alpha, beta, gamma, patience):
    if os.path.isdir(save_dir):
        raise FileExistsError(save_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(source_dir)
    os.makedirs(save_dir)

    source_config = SourceConfig.load(os.path.join(
        source_dir, 'config.json'))
    splits_file = source_config.splits_file

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
        'source_dir': source_dir,
        'name': name,
        'splits_file': splits_file,
        'n_classes': 2,
        'n_clusters': source_config.n_clusters,
        'update_inteval': 1,
        'nodes': source_config.nodes,
        'batch_size': batch_size,
        'optimizer': ('SGD', {'lr': lr, 'momentum': momentum}),
        'tol': tol,
        'maxiter': maxiter,
        'save_interval': save_interval,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'patience': patience,
    }

    ae_weights = os.path.join(source_dir, 'ae_weights.h5')
    dec_weights = os.path.join(source_dir, 'DEC_model_final.h5')
    config_args['source_weights'] = ae_weights, dec_weights

    ae_weights = os.path.join(save_dir, 'ae_weights.h5')
    dec_weights = os.path.join(save_dir, 'DEC_model_final.h5')
    config_args['save_weights'] = ae_weights, dec_weights

    config = Config(**config_args)
    config.dump()
    for i in range(2):
        shutil.copyfile(config.source_weights[i], config.save_weights[i])

    dec = MultitaskDEC(config, x_train.shape)
    dec.init(x_train)
    print(dec.model.summary())

    y_pred, metrics, best_ite = dec.clustering(
        (x_train, y_train),
        (x_train_dev, y_train_dev),
        (x_valid, y_valid))

    with open(os.path.join(save_dir, 'results_final.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'metrics': metrics,
            'best_ite': best_ite}, f)

    dec.report_run(splits)
