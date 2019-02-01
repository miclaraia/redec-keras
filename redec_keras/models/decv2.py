import os
import numpy as np
import pickle

from keras.optimizers import SGD
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import homogeneity_score

import dec_keras.DEC
from redec_keras.models.utils import get_cluster_to_label_mapping_safe, \
        calc_f1_score, one_percent_fpr
from redec_keras.models.utils import Metrics
from redec_keras.utils.pca_plot import PCA_Plot
import redec_keras.models.utils as utils


class Config(utils.Config):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_interval = kwargs.get('update_interval') or 140


class DECv2(dec_keras.DEC):

    def __init__(self, config, input_shape):
        super().__init__(
            n_clusters=config.n_clusters,
            dims=[input_shape[1]] + config.nodes,
            batch_size=config.batch_size)

        print(self.dims)

        self.n_classes = config.n_classes
        self.n_clusters = config.n_clusters
        self.config = config

    def init(self, x_train, verbose=True):
        ae_weights, dec_weights = self.config.save_weights

        self.initialize_model(
            optimizer=self.config.get_optimizer(),
            ae_weights=ae_weights,
            x=x_train)

        if os.path.isfile(dec_weights):
            self.model.load_weights(dec_weights, by_name=True)
        if verbose:
            print(self.model.summary())

    @classmethod
    def load(cls, save_dir, x_train, verbose=True):
        config = Config.load(os.path.join(save_dir, 'config.json'))
        input_shape = x_train.shape

        self = cls(config, input_shape)
        self.init(x_train, verbose)

        return self

    def calculate_metrics(self, train, test):
        c_map = self.get_cluster_map(train[0], train[1])[0]
        return self._calculate_metrics(test[0], test[1], c_map)

    def _calculate_metrics(self, x, y, c_map):
        cluster_pred = self.model.predict(x, verbose=0).argmax(1)
        #f1 = f1_score(y, np.argmax(y_pred, axis=1))
        f1c = calc_f1_score(y, cluster_pred, c_map)
        h = homogeneity_score(y, cluster_pred)
        nmi = metrics.normalized_mutual_info_score(y, cluster_pred)

        return (np.nan, f1c, h, nmi)

    def get_cluster_map(self, x, y, toprint=False):
        train_q = self.model.predict(x, verbose=0)
        # train_p = self.target_distribution(train_q)
        c_map = get_cluster_to_label_mapping_safe(
            y, train_q.argmax(1), self.n_classes, self.n_clusters,
            toprint=toprint)

        return c_map

    def clustering(
            self,
            train_data,
            train_dev_data,
            validation_data):
        x = np.concatenate((train_data[0], train_dev_data[0]))
        #y = np.concatenate((train_data[1], train_dev_data[1]))

        return super().clustering(
            x,
            tol=self.config.tol,
            maxiter=self.config.maxiter,
            update_interval=self.config.update_interval,
            save_dir=self.config.save_dir)

    def report_run(self, splits, make_samples=False):
        name = self.config.name
        save_dir = self.config.save_dir

        x_train, y_train = splits['train']
        x_train_dev, y_train_dev = splits['train_dev']
        x_test, y_test = splits['test']

        x = np.concatenate((x_train, x_train_dev))
        y = np.concatenate((y_train, y_train_dev))

        metrics = self.calculate_metrics((x, y), (x_test, y_test))

        pca_plot = PCA_Plot(self, x_train, y_train,
                            self.config.n_clusters,
                            title=name,
                            make_samples=make_samples)

        fig = pca_plot.plot()
        fig.savefig(os.path.join(save_dir, 'pca_plot.png'))
        if make_samples:
            pca_plot.plot_samples(save_dir)
        # pca_plot, pca = pca_plotv2(
            # self, x_train, y_train, self.config.n_clusters, title=name)
        # pca_plot.savefig(os.path.join(save_dir, 'pca_plot.png'))

        cmap = self.get_cluster_map(x_train, y_train)

        with open(os.path.join(self.config.save_dir, 'report.pkl'), 'wb') as f:
            pickle.dump({
                'save_dir': save_dir,
                'name': name,
                'metrics': metrics,
                'cmap': cmap}, f)
