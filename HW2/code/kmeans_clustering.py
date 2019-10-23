import numpy as np
from distance import pdist


def kmeans_clustering(all_features, vocab_size, epsilon, max_iter):
    """
    The function kmeans implements a k-means algorithm that finds the centers of vocab_size clusters
    and groups the all_features around the clusters. As an output, centroids contains a
    center of the each cluster.

    :param all_features: an N x d matrix, where d is the dimensionality of the feature representation.
    :param vocab_size: number of clusters.
    :param epsilon: When the maximum distance between previous and current centroid is less than epsilon,
        stop the iteration.
    :param max_iter: maximum iteration of the k-means algorithm.

    :return: an vocab_size x d array, where each entry is a center of the cluster.
    """

    # Your code here. You should also change the return value.
    np.random.seed(0)
    n = all_features.shape[0]
    f_n = all_features.shape[1]
    center = np.zeros((vocab_size, f_n))

    # I choose start points as random sample of all_feature
    idx = np.random.choice(n, vocab_size, replace=False)
    for i in range(vocab_size):
        center[i, :] = all_features[idx[i], :]

    # In the worst case, iterations up to max_iter
    for k in range(max_iter):

        # back-up of previous center points, because of compare of epsilon
        prev_center = np.copy(center)

        # get distance between center points and feature points and labeling of feature using np.argmin
        d = pdist(all_features, center)
        lab = np.argmin(d, axis=1)

        # For empty cluster cases, align the points in the order that is farthest from the that point's center
        mind = np.min(d, axis=1)
        sort_arg = np.argsort(mind * -1)

        # Use the labels to update the center points to the average of the points in the cluster.
        center = np.zeros((vocab_size, f_n))
        num = np.zeros(vocab_size)
        for i in range(n):
            center[lab[i], :] = center[lab[i], :] + all_features[i, :]
            num[lab[i]] += 1

        # In empty case, the empty cluster is divided by 0. Therefore, to prevent this, the point that is farthest from the center is adopted as a new center.
        id = 0
        for i in range(vocab_size):
            if num[i] == 0:
                center[i, :] = all_features[sort_arg[id], :]
                id += 1
            else:
                center[i, :] = center[i, :] / num[i]

        # The ending condition is checked by comparing the previous center point with the current point.
        d2 = pdist(prev_center, center)
        maxe = 0
        for i in range(vocab_size):
            if maxe < d2[i, i]:
                maxe = d2[i, i]

        if maxe < epsilon:
            break

    return center