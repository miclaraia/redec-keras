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
from redec_keras.utils.pca_plot import PCA_Plot
import redec_keras.models.utils
import redec_keras.models.decv2 as decv2

lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9',
            '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']

logger = logging.getLogger(__name__)

#  DEC constants from DEC paper
# batch_size = 256
# lr         = 0.01
# momentum   = 0.9
# tol        = 0.001
# maxiter    = 100
# update_interval = 140 #perhaps this should be 1 for multitask learning
# # update_interval = 10 #perhaps this should be 1 for multitask learning
# n_clusters = 10 # number of clusters to use
# n_classes  = 2  # number of classes

class Config(decv2.Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'multitask'
        self.maxiter = kwargs.get('maxiter') or 80
        self.update_interval = kwargs.get('update_interval') or 1
        self.alpha = kwargs.get('alpha') or 1.0
        self.beta = kwargs.get('beta') or 0.0
        self.gamma = kwargs.get('gamma') or 0.0
        self.patience = kwargs.get('patience') or 10


class MyLossWeightCallback(Callback):
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    # customize your behavior
    def on_epoch_begin(self, epoch, logs={}):
        self.alpha = self.alpha
        self.beta = self.beta
        self.gamma = self.gamma


class MultitaskDEC(decv2.DECv2):

    def __init__(self, config, input_shape):
        super().__init__(config, input_shape)
        self.metrics = None

    def init(self, x_train, verbose=True):
        ae_weights, dec_weights = self.config.save_weights
        print(x_train.shape)
        self.initialize_model(
            optimizer=self.config.get_optimizer(),
            ae_weights=ae_weights,
            x=x_train)

        self.model.load_weights(dec_weights, by_name=True)
        if verbose:
            print(self.model.summary())

    @classmethod
    def load(cls, save_dir, x_train, verbose=True):
        config = Config.load(save_dir)
        dec_weights = config.save_weights[1]
        input_shape = x_train.shape

        self = cls(config, input_shape)
        with open(os.path.join(save_dir, 'metrics.pkl'), 'rb') as f:
            self.metrics = pickle.load(f)

        self.init(x_train, verbose)
        self.build_model(0, 0, 0, None, None)
        self.model.load_weights(dec_weights, by_name=True)

        return self

    def calculate_metrics(self, train, test):
        c_map = self.get_cluster_map(train[0], train[1])[0]

        x, y = test
        y_pred = self.model.predict(x)[0]

        y = np_utils.to_categorical(y)
        y_pred = np_utils.to_categorical(y_pred)

        return self._calculate_metrics(x, y, y_pred, c_map)

    def _calculate_metrics(self, x, y, y_pred, c_map):
        cluster_pred = self.model.predict(x, verbose=0)[1].argmax(1)
        f1 = f1_score(y[:,1], np.argmax(y_pred, axis=1))
        f1c = calc_f1_score(y[:,1], cluster_pred, c_map)
        h = homogeneity_score(y[:,1], cluster_pred)
        nmi = metrics.normalized_mutual_info_score(y[:,1], cluster_pred)

        return (f1, f1c, h, nmi)

    def predict_clusters(self, x):
        q = self.model.predict(x)[1]
        return q.argmax(1)

    def get_cluster_map(self, x, y, toprint=False):
        if len(y.shape) == 2:
            y = y[:,1]
        q = self.model.predict(x, verbose=0)[1]
        c_map = get_cluster_to_label_mapping_safe(
            y, q.argmax(1), self.n_classes,
            self.n_clusters, toprint=toprint)
        return c_map

    def build_model(self, alpha, beta, gamma, loss, loss_weights,
                    model_1='model_1'):
        cluster_weights = self.model.get_layer(name='clustering').get_weights()

        a = Input(shape=(self.dims[0],))  # input layer

        self.model.layers[1].kernel_regularizer = regularizers.l2(0.5)
        self.model.layers[2].kernel_regularizer = regularizers.l2(0.5)
        self.model.layers[3].kernel_regularizer = regularizers.l2(0.5)
        self.model.layers[4].kernel_regularizer = regularizers.l2(0.5)

        hidden = self.encoder(a)
        q_out = ClusteringLayer(self.n_clusters, name='clustering')(hidden)

        e_out = self.autoencoder(a)

        pred = Dense(2, activation='softmax')(q_out)

        self.model = Model(inputs=a, outputs=[pred, q_out, e_out])

        self.model.get_layer(name='clustering').set_weights(cluster_weights)

        optimizer = 'adam'

        if loss is None:
            self.model.compile(
                optimizer=optimizer,
                loss={
                    'dense_1': 'categorical_crossentropy',
                    'clustering': 'kld', model_1: 'mse'},
                loss_weights={
                    'dense_1': alpha, 'clustering': beta, model_1: gamma})
        else:
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                loss_weights=loss_weights)

    def evaluate(self):
        pass

    def clustering(
            self,
            train_data,
            train_dev_data,
            validation_data,

            loss=None,
            loss_weights=None):
        train_data = (train_data[0], np_utils.to_categorical(train_data[1]))
        train_dev_data = \
            (train_dev_data[0], np_utils.to_categorical(train_dev_data[1]))
        validation_data = \
            (validation_data[0], np_utils.to_categorical(validation_data[1]))

        x, y = train_data

        print(self.config.alpha)
        alpha = K.variable(self.config.alpha)
        beta = K.variable(self.config.beta)
        gamma = K.variable(self.config.gamma)

        tol = self.config.tol
        update_interval = self.config.update_interval
        save_interval = self.config.save_interval
        maxiter = self.config.maxiter

        ae_weights, dec_weights = self.config.save_weights
        save_dir = self.config.save_dir

        best_model_fname = os.path.join(save_dir, 'best_train_dev_loss.h5')
        intermediate_model_fname = os.path.join(save_dir, 'DEC_model_{}.h5')

        if not os.path.isdir(save_dir):
            raise FileNotFoundError(
                'savedir does not exist\n{}'.format(save_dir))
        if y is None:
            logger.warn('No labels provided, won\'t print metrics')

        print('Update interval', update_interval)
        print('Save interval', save_interval)
   
        try:
            self.load_weights(dec_weights)
        except AttributeError:
            # initialize cluster centers using k-means
            print('Initializing cluster centers with k-means.')
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(self.encoder.predict(x))
            y_pred_last = y_pred
            self.model.get_layer(name='clustering') \
                .set_weights([kmeans.cluster_centers_])
   
        y_p = super().predict_clusters(x)
   
        cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
            get_cluster_to_label_mapping_safe(
                y[:,1], y_p, self.n_classes, self.n_clusters)
        
        logger.debug('')
        print(np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list)))
        cluster_to_label_mapping[np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list))] = 1
        
        ###############################################################
        ###############################################################
        ###############################################################

        logger.debug('Building Model')
        self.build_model(alpha, beta, gamma, loss, loss_weights)

        if not os.path.isdir(save_dir):
            logger.debug('Save dir doesn\'t exist')
            os.makedirs(save_dir)
   
        loss = [0, 0, 0]
        index = 0
        q = self.model.predict(x, verbose=0)[1]
        y_pred_last = q.argmax(1)
        self.metrics = Metrics()

        best_train_dev_loss = [np.inf, np.inf, np.inf]
        logger.debug('start training')

        ite = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)[1]
                valid_p = self.target_distribution(self.model.predict(validation_data[0], verbose=0)[1])
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                y_pred = self.model.predict(x)[0]
                if y is not None:
                    logger.debug('Calculating metrics')
                    c_map = get_cluster_to_label_mapping_safe(
                        y[:,1], q.argmax(1), self.n_classes,
                        self.n_clusters, toprint=False)[0]

                    metrics_train = self._calculate_metrics(x, y, y_pred, c_map)

                    x_valid = validation_data[0]
                    y_valid = validation_data[1]
                    y_pred_valid = self.model.predict(validation_data[0])[0]

                    val_loss = self.model.test_on_batch(
                        x_valid, [y_valid, valid_p, x_valid])
                    metrics_valid = self._calculate_metrics(
                        x_valid, y_valid, y_pred_valid, c_map)
                    print(self.metrics.add(
                        ite, metrics_train, metrics_valid, loss, val_loss))
                    
                    train_dev_p = self.target_distribution(
                        self.model.predict(train_dev_data[0], verbose=0)[1])
                    train_dev_loss = np.round(self.model.test_on_batch(
                        train_dev_data[0],
                        [train_dev_data[1], train_dev_p, train_dev_data[0]]
                        ), 5)
                    if train_dev_loss[1] < best_train_dev_loss[1] and \
                            train_dev_loss[-1] < best_train_dev_loss[-1]:
                            # only interested in classification improvements
                    
                        print('saving model: {} -> {}'.format(
                            best_train_dev_loss, train_dev_loss))
                        print('saving model: ', best_train_dev_loss, ' -> ', train_dev_loss)
                        self.model.save_weights(best_model_fname)
                        best_train_dev_loss = train_dev_loss
                        best_ite = ite
                        self.metrics.mark_best(ite)

                # check stop criterion
                
                if ite > 0 and delta_label < tol:
                    print('delta_label {} < tol {}'.format(delta_label, tol))
                    print('Reached tolerance threshold. Stopping training.')
                    break

                if ite - best_ite > self.config.patience:
                    print('ite-best_ite={}>{} -- Stopping'.format(
                        ite-best_ite, self.config.patience))
                    break
                
                # Classification loss
                alpha = K.variable((1 - ite/maxiter))
                # Clustering loss
                beta  = K.variable(1-alpha)  # should ignore l=this loss
                # reconstruction loss
                gamma = K.variable(1.0)
                print(K.eval(alpha), K.eval(beta), K.eval(gamma))
                logger.debug('Fitting model')
                history = self.model.fit(
                    x=x,
                    y=[y,p,x],
                    validation_data=(
                        validation_data[0], [validation_data[1], valid_p,
                        validation_data[0]]),
                    callbacks=[MyLossWeightCallback(alpha, beta, gamma)],
                    verbose=1)
                logger.debug('Done Fitting')
            else:
                print(K.eval(alpha), K.eval(beta), K.eval(gamma))
                logger.debug('Fitting model')
                history = self.model.fit(
                    x=x, y=[y,p,x],
                    validation_data=(
                        validation_data[0], [validation_data[1], valid_p,
                        validation_data[0]]),
                    verbose=1)
                logger.debug('Done Fitting')
            #history = self.model.fit(x=x, y=[y,p], callbacks=[MyLossWeightCallback(alpha, beta)], verbose=0)
            #print(history.history)
            loss = [history.history[k][0] for k in history.history.keys() if 'val' not in k]
              # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                fname = intermediate_model_fname.format(ite)
                print('saving model to: {}'.format(fname))
                self.model.save_weights(fname)

            self.metrics.save(os.path.join(
                save_dir, 'metrics.pkl'))

            ite += 1

        # save the trained model
        fname = os.path.join(
            self.config.save_dir, 'DEC_model_{}.h5'.format(ite))
        print('saving model to: {}'.format(fname))
        self.model.save_weights(fname)

        # Save best weights as DEC_model_final.h5, not current weights
        shutil.copyfile(best_model_fname, self.config.save_weights[1])

        y_p = self.model.predict(x, verbose=0)[1].argmax(1)
        cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
            get_cluster_to_label_mapping_safe(
                y[:,1], y_p, self.n_classes, self.n_clusters)
        return y_pred, self.metrics, best_ite

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

        fig = plt.figure(figsize=(15, 8))
        self.metrics.plot(fig, title=name)
        fig.savefig(os.path.join(save_dir, 'train_metrics.png'))





