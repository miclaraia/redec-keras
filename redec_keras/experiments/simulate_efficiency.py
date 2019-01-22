import numpy as np
import random
from collections import Counter

from muon.dissolving.decv2 import DECv2


class EfficiencyStudy:

    @classmethod
    def run(cls, save_dir, x, y, n_trials=5):
        dec = DECv2.load(save_dir, x)

        y_pred = dec.predict_clusters(x)
        n_classes = dec.config.n_classes
        n_clusters = dec.config.n_clusters

        total = 0
        for i in range(n_trials):
            total += cls._simulate(
                y, y_pred, n_classes, n_clusters, sample_size=25)
        mean = total / float(n_trials)
        print(mean)
        return mean

    @staticmethod
    def _simulate(y, y_pred, n_classes, n_clusters, sample_size=25):
        """
        Count how many clicks this DEC model would need to classify all subjects
        """
        click_counter = 0
        not_seen = set([x for x in range(len(y))])
        while not_seen:
            for i in range(n_clusters):
                assigned = set(np.where(y_pred == i)[0].tolist())

                assigned = assigned & not_seen
                if not assigned:
                    continue

                if sample_size < len(assigned):
                    sample = set(random.sample(assigned, sample_size))
                else:
                    sample = assigned

                not_seen = not_seen - sample
                sample = list(sample)
                mc = Counter(y[sample]).most_common()[0][0]

                n_majority = np.sum(y[sample] == mc)
                n_minority = np.sum(y[sample] != mc)
                # print('mc', mc)
                # print('sample', sample)
                # print('y', y[sample])
                # print('bool', y[sample] == mc)
                # print(n_majority, n_minority, len(sample))
                assert n_majority + n_minority == len(sample)

                click_counter += n_minority
        print(click_counter)
        return click_counter

