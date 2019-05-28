import numpy as np
import random
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class EfficiencyStudy:

    @classmethod
    def run(cls, dec, x, y, n_trials=5, sample_size=25):
        # dec = DECv2.load(save_dir, x)

        y_pred = dec.predict_clusters(x)
        n_classes = dec.config.n_classes
        n_clusters = dec.config.n_clusters

        return cls._simulate_trials(
            y, y_pred, n_trials, n_classes,
            n_clusters, sample_size=sample_size)

    @classmethod
    def _simulate_trials(cls, y, y_pred, n_trials, n_classes,
                         n_clusters, sample_size):
        trials = []
        for _ in range(n_trials):
            trials.append(cls._simulate(
                y, y_pred, n_classes, n_clusters, sample_size=sample_size))
            logger.info(trials[-1])

        return np.mean(trials), np.std(trials, ddof=1)

    @staticmethod
    def _simulate(y, y_pred, n_classes, n_clusters, sample_size=25):
        """
        Count how many clicks this DEC model would need to classify all subjects
        """
        click_counter = 0

        for i in range(n_clusters):
            cluster_subjects = np.where(y_pred == i)[0]
            if len(cluster_subjects) == 0:
                continue

            cluster_permutation = np.random.permutation(cluster_subjects)

            n = len(cluster_permutation) % sample_size
            diff = sample_size - n
            if n > 0:
                cluster_permutation = np.concatenate((
                    cluster_permutation,
                    np.random.choice(cluster_permutation[:-n], diff)))

            j = 0
            while j < len(cluster_permutation):
                sample = y[cluster_permutation[j:j+sample_size]]
                labels, counts = np.unique(sample, return_counts=True)

                j += sample_size

                # print(counts)
                if min(counts) < sample_size:
                    click_counter += min(counts)
        print(click_counter)
        print(np.unique(y, return_counts=True))
        return click_counter

