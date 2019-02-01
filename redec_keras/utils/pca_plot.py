import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import numpy as np

from sklearn.decomposition import PCA


class PCA_Plot:

    def __init__(self, dec, x, y, n_clusters, title, make_samples=False):
        self.x = x
        self.y = y
        self.y_pred = dec.predict_clusters(x)
        self.n_clusters = n_clusters
        self.title = title
        self.ccolor = '#4D6CFA'

        self.pca = None
        self.x_pca = None
        self.y_pca = None
        self.c_pca = None

        self.cluster_centers = None
        self.cluster_ids = None
        self.labels = None

        self.samples = None

        self._init_cluster_centers(dec)
        self._init_pca(dec)
        if make_samples:
            self._init_samples()

    def _init_cluster_centers(self, dec):
        clustering_layer = dec.model.get_layer(name='clustering')
        weights = clustering_layer.get_weights()
        cluster_centers = np.squeeze(np.array(weights))

        unique = np.unique(self.y_pred)
        cluster_centers = cluster_centers[unique,:]

        labels = [str(i) for i in range(self.n_clusters)]
        labels = np.array(labels)[unique]

        self.cluster_centers = cluster_centers
        self.cluster_ids = unique
        self.labels = labels

    def _init_pca(self, dec):
        base_network = dec.encoder

        pca = PCA(n_components=3)
        pca_xy = pca.fit_transform(base_network.predict(self.x))
        print(pca.explained_variance_ratio_)
        pca_centers = pca.transform(self.cluster_centers)

        self.pca = pca
        self.x_pca = pca_xy[:, 0]
        self.y_pca = pca_xy[:, 1]
        self.c_pca = pca_centers

    def plot(self):
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        samples = self.samples is not None
        self._plot_ax(ax1, self.y.astype(np.int), 'Color by Class',
                     samples=samples)
        self._plot_ax(ax2, self.y_pred, 'Color by Cluster')

        fig.suptitle(self.title)
        return fig

    def _plot_ax(self, ax, y, subtitle, samples=False):
        unique_targets = list(np.unique(y))

        N = len(unique_targets)
        cmap = discrete_cmap(N, 'jet')
        norm = matplotlib.colors.BoundaryNorm(
            np.arange(0, max(3, N), 1), max(3, N))

        if -1 in unique_targets:
            unique_targets.remove(-1)

        for i, l in enumerate(unique_targets):
            _x = self.x_pca[np.where(y == l)]
            _y = self.y_pca[np.where(y == l)]
            _c = i * np.ones(_x.shape)
            ax.scatter(_x, _y, marker='o', s=5, c=_c,
                       cmap=cmap, norm=norm, alpha=0.2,
                       label=self.labels[np.where(self.cluster_ids==l)])
                       # picker=True)

        if samples:
            self._plot_sample_outlines(ax)

        ax.scatter(
            self.c_pca[:,0], self.c_pca[:,1],
            marker='o',
            s=40,
            color=self.ccolor,
            alpha=1.0,
            label='cluster centre')

        for i in range(len(self.cluster_centers)):
            ax.text(self.c_pca[i,0], self.c_pca[i,1], str(i), size=20,
                    path_effects=[
                        PathEffects.withStroke(linewidth=3, foreground="w")])
        ax.set_title(subtitle)    
        ax.set_axis_off()

    def _init_samples(self):
        samples = []
        y = self.y_pred
        for l in list(np.unique(y)):
            i = np.where(y==l)[0]
            if len(i) > 0:
                i = np.random.choice(i, 1)[0]
                samples.append((l, i))

        self.samples = samples

    def _plot_sample_outlines(self, ax):
        for l, i in self.samples:
            x, y = self.x_pca[i], self.y_pca[i]
            rect = patches.Rectangle(
                (x-2, y-2), 4, 4,
                linewidth=1, edgecolor='k',
                facecolor='none')
            print(l, x, y, rect)
            ax.add_patch(rect)

    def plot_samples(self, save_dir):
        pass
        # for l, i in self.samples:
            # fname = os.path.join(save_dir, 'sample_{}.png'.format(l))
            # print(l, i, fname)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)

            # print(self.x[i])
            # print(self.x.shape, i)
            # data = Camera().transform(self.x[i,:], False)
            # print(data)
            # CameraPlot.plot(data, ax)

            # fig.savefig(fname)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap \
        .from_list(cmap_name, color_list, N)
