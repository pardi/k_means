import logging

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List

import imageio.v2 as imageio

logging.basicConfig(level=logging.INFO)

def make_gif():

    # build gif
    # with imageio.get_writer('bayesian_opt.gif', mode='I') as writer:
    images = []
    for idx in range(100):
        filename = str(idx) + '.png'
        images.append(imageio.imread(filename))
        # writer.append_data(image)

    imageio.mimsave('bayesian_opt.gif', images)

@dataclass
class Cluster:
    mean: np.array
    name: str
    samples: list

def empty_clusters(clusters: List[Cluster]):

    for cluster in clusters:
        cluster.samples = []

    return clusters

def initialise_clusters(clusters: List[Cluster]):

    for cluster in clusters:
        cluster.mean = np.random.rand(2)

    return clusters


def single_assignment_step(sample: np.array, clusters: List[Cluster]):

    means = [cluster.mean for cluster in clusters]

    nearest_cluster_idx = np.argmin([np.linalg.norm(x) for x in sample - means])

    return nearest_cluster_idx


def assignment_step(samples: np.array, clusters: List[Cluster]):

    empty_clusters(clusters=clusters)

    for sample in samples.transpose():
        nearest_cluster_idx = single_assignment_step(sample, clusters)
        clusters[nearest_cluster_idx].samples.append(sample)

    return clusters


def update_step(clusters: List[Cluster]):
    for cluster in clusters:
        if cluster.samples:
            cluster.mean = np.mean(cluster.samples, axis=0)

    return clusters


def error_clusters(clusters: List[Cluster], old_error: float = 0):
    error = np.sum([np.linalg.norm(cluster.mean) for cluster in clusters])

    if abs(error - old_error) < 0.01:
        stop = True
    else:
        stop = False
    return stop, error

def compute_angles(clusters: List[Cluster]):
    angles = []

    means = [cluster.mean for cluster in clusters]

    mean_centroid = np.mean(means, axis=0)

    for cluster in clusters:
        vec = cluster.mean - mean_centroid
        angles.append(np.arctan2(vec[1], vec[0]))

    return angles, mean_centroid

def sample_mult_distr(means: np.array, stds: np.array):

    distr_idx = np.random.choice(len(means))

    return np.random.normal(means[distr_idx], stds[distr_idx], size=(2, 1))

# Lloyd's algorithm
def naive_kmeans():
    #
    distr_means = np.random.rand(2) - .5
    distr_stds = .1 * np.random.rand(2)

    old_error = 1e10

    clusters = [Cluster(np.array([0.2, 0.2]), "K1", []), Cluster(np.array([-0.2, 0.2]), "K2", [])]
    clusters = initialise_clusters(clusters)
    samples = sample_mult_distr(distr_means, distr_stds)

    for _ in range(10):
        sample = sample_mult_distr(distr_means, distr_stds)

        samples = np.concatenate((samples, sample), axis=1)

        clusters = assignment_step(samples, clusters)

        clusters = update_step(clusters)

        stop, old_error = error_clusters(clusters, old_error=old_error)

        distr_angles, mean_centroid = compute_angles(clusters)

        angles = distr_angles[0]

        plt.plot(samples[0], samples[1], 'o')
        plt.plot(mean_centroid[0], mean_centroid[1], 'ok')
        plt.plot([mean_centroid[0], mean_centroid[0] + np.cos(angles[0])],
                 [mean_centroid[1], mean_centroid[1] + np.sin(angles[0])], 'k')
        plt.plot([mean_centroid[0], mean_centroid[0] + np.cos(angles[1])],
                 [mean_centroid[1], mean_centroid[1] + np.sin(angles[1])], 'k')

        plt.show()

        print(old_error)

    a = 0

def kmeans():

    stds = .1 * np.random.rand(2)
    means = np.random.rand(2) - .5

    X = []
    X.append(np.random.normal(means[0], stds[0], size=(2, 100)))
    X.append(np.random.normal(means[1], stds[1], size=(2, 100)))

    plt.xlim([-0.55, 0.55])
    plt.ylim([-0.55, 0.55])

    plt.plot(X[0][0, :], X[0][1, :], 'o')
    plt.plot(X[1][0][:], X[1][1][:], 'x')
    plt.grid()


def test_assignment():
    naive_kmeans()

if __name__ == "__main__":
    test_assignment()
    plt.show()