import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pickle
import json
import os
import csv
import pandas
from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_curve, homogeneity_score
from sklearn import metrics

from keras.utils import np_utils
from keras.optimizers import SGD

from dec_keras.DEC import DEC, ClusteringLayer, cluster_acc


class Config:
    def __init__(
            self, save_dir, splits_file, source_dir,
            source_weights, save_weights,
            **kwargs):
        self.save_dir  = save_dir
        self.name = kwargs.get('name')
        self.n_classes = kwargs.get('n_classes') or 2
        self.n_clusters = kwargs.get('n_clusters') or 10
        self.batch_size = kwargs.get('batch_size') or 256
        self.nodes = kwargs.get('nodes') or [500, 500, 2000, 10]
        self.optimizer = kwargs.get('optimizer') or \
            ('SGD', {'lr': .01, 'momentum': .9})
        self.tol = kwargs.get('tol') or .001
        self.maxiter = kwargs.get('maxiter') or 80
        self.save_interval = kwargs.get('save_interval') or 5
        self.update_interval = kwargs.get('update_interval') or 140

        self.splits_file = splits_file
        self.source_dir = source_dir
        self.source_weights = source_weights
        self.save_weights = save_weights

    def get_optimizer(self):
        optimizer, kwargs = self.optimizer
        print(optimizer, kwargs)

        optimizer = {
            'SGD': lambda kwargs: SGD(**kwargs)
        }[optimizer](kwargs)

        return optimizer

    def dump(self):
        fname = os.path.join(self.save_dir, 'config.json')
        json.dump(self.__dict__, open(fname, 'w'))

    @classmethod
    def load(cls, fname):
        data = json.load(open(fname, 'r'))
        if 'source_dir' not in data:
            data['source_dir'] = os.path.dirname(data['source_weights'][0])
        return cls(**data)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


class MetricItem:
    def __init__(self, f1, f1c, h, nmi, loss, type_):
        if type(loss) is not list:
            loss = [loss]
        self.f1 = f1
        self.f1c = f1c
        self.h = h
        self.nmi = nmi
        self.loss = loss
        self.type = type_

    def dump(self, with_type=False):
        if with_type:
            return OrderedDict([
                ('{}_f1'.format(self.type), self.f1),
                ('{}_f1c'.format(self.type), self.f1c),
                ('{}_h'.format(self.type), self.h),
                ('{}_nmi'.format(self.type), self.nmi)])
        else:
            return OrderedDict([
                ('f1', self.f1),
                ('f1c', self.f1c),
                ('h', self.h),
                ('nmi', self.nmi)])

    def __str__(self):
        s = '{:5s} F1={:.4f} F1c={:.4f} h={:.4f} nmi={:.4f} loss={}'
        loss = '[{}]'.format(','.join(['{:.2f}'.format(l) for l in self.loss]))
    
        return s.format(self.type, self.f1, self.f1c, self.h, self.nmi, loss)


class Metrics:
    _metric_names = ['f1', 'f1c', 'h', 'nmi']

    def __init__(self):
        self.metrics = []
        self.best_ite = None

    def _print(self, metric):

        if type(metric['iteration']) is int:
            s = '-------------------------------------------------\n' \
                '{:4d} {}\n' \
                '     {}\n' \
                '     Best Iteration: {}\n' \
                '-------------------------------------------------\n'
        else:
            s = '-------------------------------------------------\n' \
                '{:3.3f} {}\n' \
                '       {}\n' \
                '       Best Iteration: {}\n' \
                '-------------------------------------------------\n'

        # s = '-------------------------------------------------\n' \
            # '%4d  F1=%.4f  F1c=%.4f  h=%.4f  nmi=%.4f\n' \
            # '     vF1=%.4f vF1c=%.4f vh=%.4f vnmi=%.4f\n' \
            # '      loss=%s\n' \
            # '     vloss=%s\n' \
            # '-------------------------------------------------\n'

        return s.format(metric['iteration'], metric['train'], metric['valid'],
                        self.best_ite)

    def get_ite(self, iteration):
        for item in self.metrics:
            if item['iteration'] == iteration:
                return item

    def print_ite(self, iteration):
        self._print(self.get_ite(iteration))

    def print_last(self):
        self._print(self.metrics[-1])

    def last_ite(self):
        return self.metrics[-1]['iteration']

    def dump(self):
        data = []
        for m in self.metrics:
            item = OrderedDict([('iteration', m['iteration'])])
            item.update(m['train'].dump(with_type=True))
            item.update(m['valid'].dump(with_type=True))
            data.append(item)
        return data

    def get_best(self):
        return self.get_ite(self.best_ite)

    def mark_best(self, ite):
        self.best_ite = ite

    def add(self, iteration, train, valid, loss, val_loss):
        train = MetricItem(*train, loss, type_='train')
        valid = MetricItem(*valid, val_loss, type_='valid')
        metric = {
            'iteration': iteration,
            'train': train,
            'valid': valid
        }
        self.metrics.append(metric)
        return self._print(metric)

    def zip(self):
        data = {k: [] for k in ['iteration'] + self._metric_names}
        for m in self.metrics:
            data['iteration'].append(m['iteration'])
            train = m['train'].dump()
            valid = m['valid'].dump()
            for k in ['f1', 'f1c', 'h', 'nmi']:
                data[k].append((train[k], valid[k]))

        return data

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def _plot_one(self, ax, key, title=None, standard_metrics=None):
        data = self.zip()
        key1 = 'train_{}'.format(key)
        key2 = 'valid_{}'.format(key)

        x = data['iteration']
        y1, y2 = zip(*data[key])

        ax.plot(x, y1)
        ax.plot(x, y2)

        ax.legend([key1, key2])
        ax.set_ylim((0,1))

        if standard_metrics:
            standard_metrics.plot(ax, key)
        ax.set_title(title or key)

    def plot(self, fig, title=None, standard_metrics=None, keys=None):
        keys = keys or self._metric_names
        if len(keys) == 1:
            subs = 111
        else:
            subs = 2, np.ceil(len(keys)/2).astype(int), 1
            subs = sum([10**(2-i)*subs[i] for i in range(3)])

        for i, key in enumerate(keys):
            ax = fig.add_subplot(subs + i)
            self._plot_one(ax, key, None, standard_metrics)

        fig.suptitle(title)
        return fig


class MetricsCombined:

    def __init__(self, multitask, redec):
        self.multitask = multitask
        self.redec = redec

    def plot(self, fig, title=None):
        c = ['#1f77b4', '#ff7f0e']
        data_m = self.multitask.zip()
        data_r = self.redec.zip()
        for i, key in enumerate(self.multitask._metric_names):
            key1 = 'train_{}'.format(key)
            key2 = 'valid_{}'.format(key)

            ax = fig.add_subplot(221+i)

            # multitask
            x = np.array(data_m['iteration'])
            i = np.where(x==self.multitask.best_ite)[0][0]
            x_gap = x[i]

            y1, y2 = zip(*data_m[key])
            y1 = np.array(y1)
            y2 = np.array(y2)

            ax.plot(x[:i], y1[:i], c[0])
            ax.plot(x[:i], y2[:i], c[1])

            ax.plot(x[i:], y1[i:], c[0], alpha=.3)
            ax.plot(x[i:], y2[i:], c[1], alpha=.3)

            # redec
            x = np.array(data_r['iteration'])
            x += self.multitask.best_ite
            a = np.ones(x.shape)

            y1, y2 = zip(*data_r[key])
            y1 = np.array(y1)
            y2 = np.array(y2)

            ax.plot(x, y1, c=c[0])
            ax.plot(x, y2, c=c[1])
            ax.plot([x_gap, x_gap], [0, 1], '--')

            ax.legend([key1, key2])
            ax.set_title(key)

        fig.suptitle(title)
        return fig


class StandardMetrics:

    def __init__(self):
        self.metrics = {}
        self.fields = ['name', 'f1', 'h', 'nmi', 'precision', 'recall']

    def add(self, name, y, y_pred):
        item = self.calculate_metrics(y, y_pred, name)
        self.metrics[name] = item

    def dump(self, fname):
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            for name in self.metrics:
                writer.writerow(self.metrics[name])

    def to_df(self):
        return pandas.DataFrame(list(self.metrics.values()), columns=self.fields)

    def __str__(self):
        return str(self.to_df())

    @staticmethod
    def precision(y, y_pred):
        #tp/(tp+fp)
        i = np.where(y_pred==1)
        return sum(y[i])/sum(y_pred)

    @staticmethod
    def recall(y, y_pred):
        # tp/(tp+fn)
        i = np.where(y_pred==1)
        return sum(y[i]) / sum(y)
        
    @classmethod
    def calculate_metrics(cls, y, y_pred, name):
        f1 = f1_score(y, y_pred)
        h = homogeneity_score(y, y_pred)
        nmi = metrics.normalized_mutual_info_score(y, y_pred)
        precision = cls.precision(y, y_pred)
        recall = cls.recall(y, y_pred)

        return {
            'name': name, 'f1': f1, 'h': h, 'nmi': nmi,
            'precision': precision, 'recall': recall}
    
    def plot(self, ax, key):
        if key == 'f1c':
            key = 'f1'
        if key in self.fields:
            l = list(self.metrics)
            y = [self.metrics[i][key] for i in l]

            ax2 = ax.twinx()
            ax2.set_yticks(y)
            ax2.set_yticklabels(l)

            for _y in y:
                ax2.axhline(_y, linestyle='--', alpha=.5)

            ax2.set_ylim(ax.get_ylim())


def one_percent_fpr(y, pred, fom):
    fpr, tpr, thresholds = roc_curve(y, pred)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
    return FoM, threshold, fpr, tpr


def calc_f1_score(y_true, predicted_clusters, cluster_to_label_mapping):
    y_pred = []
    for i in range(len(y_true)):
          y_pred.append(cluster_to_label_mapping[predicted_clusters[i]])
    return f1_score(y_true, np.array(y_pred))


def get_cluster_to_label_mapping_safe(y, y_pred, n_classes, n_clusters, toprint=True):
    """Enusre at least one cluster assigned to each label.
    """
    one_hot_encoded = np_utils.to_categorical(y, n_classes)

    cluster_to_label_mapping = []
    n_assigned_list = []
    majority_class_fractions = []
    majority_class_pred = np.zeros(y.shape)
    for cluster in range(n_clusters):
        cluster_indices = np.where(y_pred == cluster)[0]
        n_assigned_examples = cluster_indices.shape[0]
        n_assigned_list.append(n_assigned_examples)
        cluster_labels = one_hot_encoded[cluster_indices]
        cluster_label_fractions = np.mean(cluster_labels, axis=0)
        majority_cluster_class = np.argmax(cluster_label_fractions)
        cluster_to_label_mapping.append(majority_cluster_class)
        majority_class_pred[cluster_indices] += majority_cluster_class
        majority_class_fractions.append(cluster_label_fractions[majority_cluster_class])
        if toprint:
            print(cluster, n_assigned_examples, majority_cluster_class, cluster_label_fractions[majority_cluster_class])
    #print(cluster_to_label_mapping)
    if toprint:
        print(np.unique(y), np.unique(cluster_to_label_mapping))
    try:
        # make sure there is at least 1 cluster representing each class
        assert np.all(np.unique(y) == np.unique(cluster_to_label_mapping))
    except AssertionError:
        # if there is no cluster for a class then we will assign a cluster to that
        # class

        # find which class it is
        # ASSUMPTION - this task is binary

        diff = list(set(np.unique(y)) - set(np.unique(cluster_to_label_mapping)))[0]
          # we choose the cluster that contains the most examples of the class with no cluster

        one_hot = np_utils.to_categorical(y_pred[np.where(y==diff)[0]], \
                                            len(cluster_to_label_mapping))

        cluster_to_label_mapping[np.argmax(np.sum(one_hot, axis=0))] = int(diff)
    if toprint:
        print(cluster_to_label_mapping)
    return cluster_to_label_mapping, n_assigned_list, majority_class_fractions


def get_cluster_centres(dec):
    return np.squeeze(np.array(
        dec.model.get_layer(name='clustering').get_weights()))


# def pca_plotv2(dec, x, y, n_clusters, title):
    # y_pred = dec.predict_clusters(x)

    # cluster_centers = dec.model.get_layer(name='clustering')
    # cluster_centers = cluster_centers.get_weights()
    # cluster_centers = np.squeeze(np.array(cluster_centers))

    # labels = [str(i) for i in range(n_clusters)]
    
    # unique = np.unique(y_pred)
    # cluster_ids = unique
    # cluster_centers = cluster_centers[unique,:]
    # labels = np.array(labels)[unique]
    
    # return _pca_plotv2(
        # dec, x, y, y_pred, cluster_centers, cluster_ids, labels=labels, title=title)


# def _pca_plotv2(dec, x, y, y_pred, cluster_centres, cluster_ids, labels,
                # ulcolour='#747777', ccolour='#4D6CFA', title=None):
    # base_network = dec.encoder

    # pca = PCA(n_components=3)
    # x_pca = pca.fit_transform(base_network.predict(x))
    # print(pca.explained_variance_ratio_)
    # c_pca = pca.transform(cluster_centres)

    # fig = plt.figure(figsize=(16, 8))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    
    # ax = [ax1, ax2]
    # y = [y.astype(np.int), y_pred]
    # subtitle = ['Color by Class', 'Color by Cluster']
    
    # for ax, y, subtitle in zip(ax, y, subtitle):
        # unique_targets = list(np.unique(y))

        # N = len(unique_targets)
        # cmap = discrete_cmap(N, 'jet')
        # norm = matplotlib.colors.BoundaryNorm(
            # np.arange(0, max(3, N), 1), max(3, N))

        # if -1 in unique_targets:
            # unique_targets.remove(-1)
        # for i, l in enumerate(unique_targets):
            # _x = x_pca[np.where(y == l), 0]
            # _y = x_pca[np.where(y == l), 1]
            # _c = i * np.ones(_x.shape)
            # ax.scatter(_x, _y, marker='o', s=5, c=_c,
                       # cmap=cmap, norm=norm, alpha=0.2, label=labels[np.where(cluster_ids==l)])
            
        # ax.scatter(
            # c_pca[:,0], c_pca[:,1],
            # marker='o', s=40, color=ccolour, alpha=1.0, label='cluster centre')
        # for i in range(len(cluster_centres)):
            # ax.text(c_pca[i,0], c_pca[i,1], str(i), size=20)
        # ax.set_title(subtitle)    
        # ax.set_axis_off()

    # fig.suptitle(title)
    # return fig, pca


# def pca_plot(base_network, x, cluster_centres, y=None, labels=[],
             # lcolours=[], ulcolour='#747777', ccolour='#4D6CFA'):
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(np.nan_to_num(base_network.predict(x)))
    # c_pca = pca.transform(cluster_centres)
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111)
    # if np.any(y):
        # unique_targets = list(np.unique(y))
        # if -1 in unique_targets:
            # ax.scatter(
                # x_pca[np.where(y == -1), 0],
                # x_pca[np.where(y == -1), 1],
                # marker='o',
                # s=20,
                # color=ulcolour, alpha=0.1)
            # unique_targets.remove(-1)
        # for l in unique_targets:
            # l = int(l)
            # ax.scatter(
                # x_pca[np.where(y == l), 0],
                # x_pca[np.where(y == l), 1],
                # marker='o', s=5,
                # color=lcolours[l],
                # alpha=0.7,
                # label=labels[l])
    # else:
        # ax.scatter(
            # x_pca[:,0], x_pca[:,1],
            # marker='o', s=20, color=ulcolour, alpha=0.7)
    # ax.scatter(
        # c_pca[:,0], c_pca[:,1],
        # marker='o', s=40, color=ccolour, alpha=1.0, label='cluster centre')

    # for i in range(len(cluster_centres)):
        # ax.text(c_pca[i,0], c_pca[i,1], str(i), size=20)
    # plt.axis('off')
    # #plt.legend(ncol=2)
    # plt.show()


# def discrete_cmap(N, base_cmap=None):
    # """Create an N-bin discrete colormap from the specified input map"""

    # # Note that if base_cmap is a string or None, you can simply do
    # #    return plt.cm.get_cmap(base_cmap, N)
    # # The following works for string, None, or a colormap instance:

    # base = plt.cm.get_cmap(base_cmap)
    # color_list = base(np.linspace(0, 1, N))
    # cmap_name = base.name + str(N)
    # return matplotlib.colors.LinearSegmentedColormap \
            # .from_list(cmap_name, color_list, N)
