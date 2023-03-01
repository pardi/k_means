import logging

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from cluster import Cluster
from sampler import Sampler

logging.basicConfig(level=logging.INFO)

def assignment_step(samples: np.array, clusters: List[Cluster]):

    # Tidy up cluster and obtain mean of each cluster
    clusters_mean = []

    for cluster in clusters:
        cluster.empty()
        clusters_mean.append(cluster.mean)

    clusters_mean = np.array(clusters_mean).reshape(len(clusters), -1)

    for sample in samples.transpose():
        sample = sample.reshape(1, -1)

        dist_from_clusters = np.linalg.norm(sample - clusters_mean, axis=1)

        idx_nearest_cluster = np.argmin(dist_from_clusters)

        clusters[idx_nearest_cluster].samples.append(sample)

    return clusters


def update_step(clusters: List[Cluster]):
    for cluster in clusters:
        cluster.update_mean()
    return clusters


def error_clusters(clusters: List[Cluster], old_error: float = 0):
    error = np.sum([np.linalg.norm(cluster.mean) for cluster in clusters])

    if abs(error - old_error) < 0.01:
        stop = True
    else:
        stop = False
    return stop, error


def compute_sectors(clusters: List[Cluster]):
    angles = []

    means = [cluster.mean for cluster in clusters]

    mean_centroid = np.mean(means, axis=0)

    for cluster in clusters:
        vec = cluster.mean - mean_centroid
        angles.append(float(np.arctan2(vec[1], vec[0])))

    angles.sort()
    return mean_centroid, angles


# Lloyd's algorithm
def naive_kmeans(k_dim: int = 3, dim=2):

    cols_dict = {0: 'b',
                 1: 'r',
                 2: 'g',
                 3: 'y',
                 4: 'k'}

    # Create clusters
    clusters = []
    for idx_cluster in range(k_dim):
        clusters.append(Cluster(col=cols_dict[idx_cluster]))

    # Create sample dataset
    sample_dataset = {}
    for idx_dim in range(k_dim):
        sample_dataset[idx_dim] = []

    sampler = Sampler(num_distributions=k_dim, dim=dim)

    samples, idx_distribution = next(sampler())
    sample_dataset[idx_distribution].append(samples)

    for _, (sample, idx_distribution) in zip(range(100), sampler()):

        sample_dataset[idx_distribution].append(sample)

        samples = np.concatenate((samples, sample), axis=1)

        clusters = assignment_step(samples, clusters)

        clusters = update_step(clusters)


    # PLOT

    mean_centroid, polarity_angles = compute_sectors(clusters)

    mid_angles = []
    for idx_cluster in range(k_dim - 1):
        mid_angles.append((polarity_angles[idx_cluster] + polarity_angles[idx_cluster + 1]) / 2.0)
    mid_angles.append((polarity_angles[-1] + polarity_angles[0] + 2.0 * np.pi) / 2.0)

    # plot limits
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    # Plot point distributions
    for idx_cluster in range(k_dim):
        size_specific_samples = len(sample_dataset[idx_cluster])
        specific_samples = np.zeros((dim, size_specific_samples))

        for idx_sample, sample in enumerate(sample_dataset[idx_cluster]):
            if specific_samples.shape[1] == 1:
                specific_samples = sample
            else:
                for idx_element in range(sample.shape[0]):
                    specific_samples[idx_element, idx_sample] = sample[idx_element]

        if size_specific_samples != 0:
            if size_specific_samples == 1:
                plt.plot(specific_samples[0], specific_samples[1], 'o', color=clusters[idx_cluster].col)
            else:
                plt.plot(specific_samples[0, :], specific_samples[1, :], 'o', color=clusters[idx_cluster].col)

    # Plot cluster centroid
    for cluster in clusters:
        plt.plot(cluster.mean[0], cluster.mean[1], 'x')

    # Plot line for splitting areas
    plt.plot(mean_centroid[0], mean_centroid[1], 'ok')

    for mid_angle in mid_angles:
        plt.plot([mean_centroid[0], mean_centroid[0] + 3 * np.cos(mid_angle)],
                 [mean_centroid[1], mean_centroid[1] + 3 * np.sin(mid_angle)], 'k')

    plt.grid()

    plt.show()


if __name__ == "__main__":
    naive_kmeans()
