import numpy as np
import os
import logging
import shutil
# from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import homogeneity_score

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LambdaCallback
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense
# from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras import backend as K
# from keras.optimizers import SGD
from keras import regularizers
from keras.utils import np_utils
# from keras.models import load_model

from dec_keras.DEC import DEC, ClusteringLayer
from redec_keras.models.utils import get_cluster_to_label_mapping_safe, \
        calc_f1_score, one_percent_fpr
from redec_keras.models.utils import Metrics
from redec_keras.models.utils import pca_plotv2
import redec_keras.models.utils
import redec_keras.models.decv2 as decv2


class Config(redec_keras.models.utils.Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss = kwargs.get('loss') or 'categorical_crossentropy'
        self.patience = kwargs.get('patience') or 10


class Supervised:

    def __init__(self, config, input_shape):
        self.config = config
        self.input_shape = input_shape
        self.model = None
        self.metrics = None

    def init(self, verbose=True):
        save_weights = self.config.save_weights

        self.model = self.build_model()
        if os.path.isfile(save_weights):
            self.model.load_weights(save_weights, by_name=True)
        if verbose:
            print(self.model.summary())

    def build_model(self):
        optimizer = self.config.get_optimizer()
        loss = self.config.loss

        model = Sequential()
        model.add(Dense(self.config.nodes[0], activation='relu',
                        input_dim=self.input_shape[1]))
        model.add(Dense(self.config.nodes[1], activation='relu'))
        model.add(Dense(self.config.nodes[2], activation='relu'))
        model.add(Dense(self.config.nodes[3], activation='relu'))
        model.add(Dense(self.config.n_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer)

        if os.path.isfile(self.config.save_weights):
            model.load_weights(self.config.save_weights)

        return model

    @classmethod
    def load(cls, save_dir, x_train, verbose=True):
        config = Config.load(os.path.join(save_dir, 'config.json'))
        input_shape = x_train.shape

        self = cls(config, input_shape)
        with open(os.path.join(save_dir, 'metrics.pkl'), 'rb') as f:
            self.metrics = pickle.load(f)
        self.init(verbose)

        return self

    def calculate_metrics(self, train, test):
        x = test[0]
        y = test[1]
        y_pred = self.model.predict(x)
        return self._calculate_metrics(x, y, y_pred)

    def _calculate_metrics(self, x, y, y_pred):
        f1 = f1_score(y, np.argmax(y_pred, axis=1))

        return (f1, np.nan, np.nan, np.nan)

    def train(
            self,
            train_data,
            train_dev_data,
            validation_data):
        x = np.concatenate((train_data[0], train_dev_data[0]))
        y = np.concatenate((train_data[1], train_dev_data[1]))
        y = np_utils.to_categorical(y, 2)

        x_valid, y_valid = validation_data
        y_valid = np_utils.to_categorical(y_valid, 2)

        self.metrics = Metrics()

        epochs = self.config.maxiter
        patience = self.config.patience
        batch_size = self.config.batch_size

        def metrics_callback(epoch, logs):
            def calculate(x, y):
                y_pred = self.model.predict(x)
                return self._calculate_metrics(x, y[:,1], y_pred)

            train = calculate(x, y)
            valid = calculate(x_valid, y_valid)
            loss = [logs['loss']]
            val_loss = [logs['val_loss']]

            print(self.metrics.add(epoch, train, valid, loss, val_loss))

        callbacks = [
            ModelCheckpoint(
                self.config.save_weights, monitor='val_loss',
                save_best_only=True, mode='min'),
            EarlyStopping(
                monitor='val_loss', min_delta=0, patience=patience,
                verbose=0, mode='min'),
            LambdaCallback(on_epoch_end=metrics_callback)
        ]

        self.model.fit(
            x, y,
            validation_data=(x_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

        fname = os.path.join(self.config.save_dir, 'last_weights.h5')
        print('saving model to {}'.format(fname))
        self.model.save_weights(fname)

        print('best model at {}'.format(self.config.save_weights))

    # def clustering(
            # self,
            # train_data,
            # train_dev_data,
            # validation_data):
        # x = np.concatenate((train_data[0], train_dev_data[0]))
        # y = np.concatenate((train_data[1], train_dev_data[1]))



        # callbacks = [
            # ModelCheckpoint(
                # self.config.checkpoint, monitor='val_loss',
                # save_best_only=True, mode='min'),
            # EarlyStopping(
                # monitor='val_loss', min_delta=0, patience=patience,
                # verbose=0, mode='min'),
        # ]

        # model = self.model
        # model.fit(
            # x, y,
            # validation_data=(valid_x, valid_y),
            # epochs=epochs,
            # batch_size=batch_size,
            # callbacks=callbacks)

        # return super().clustering(
            # x,
            # tol=self.config.tol,
            # maxiter=self.config.maxiter,
            # update_interval=self.config.update_interval,
            # save_dir=self.config.save_dir)

    def report_run(self, splits):
        name = self.config.name
        save_dir = self.config.save_dir

        x_train, y_train = splits['train']
        x_train_dev, y_train_dev = splits['train_dev']
        x_test, y_test = splits['test']

        x = np.concatenate((x_train, x_train_dev))
        y = np.concatenate((y_train, y_train_dev))

        metrics = self.calculate_metrics((x, y), (x_test, y_test))

        # pca_plot, pca = pca_plotv2(
            # self, x_train, y_train, self.config.n_clusters, title=name)
        # pca_plot.savefig(os.path.join(save_dir, 'pca_plot.png'))

        # cmap = self.get_cluster_map(x_train, y_train)

        with open(os.path.join(self.config.save_dir, 'report.pkl'), 'wb') as f:
            pickle.dump({
                'save_dir': save_dir,
                'name': name,
                'metrics': metrics}, f)

        fig = plt.figure(figsize=(15, 8))
        self.metrics.plot(fig, title=name)
        fig.savefig(os.path.join(save_dir, 'train_metrics.png'))
        with open(os.path.join(
                self.config.save_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metrics, f)

