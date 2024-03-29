import cv2
import numpy as np
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction


def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab.mat' exists and contains an N x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]
    N = len(image_paths)
    hist = np.zeros((N, vocab_size))
    # Your code here. You should also change the return value.
    k = 0
    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]

        # get features of image
        features = feature_extraction(img, feature)

        # get distance of features and codevectors, then get histogram
        d = pdist(features, vocab);
        lab = np.argmin(d, axis=1)
        for l in lab:
            hist[k, l] += 1
        hist[k,:] = hist[k,:] / len(lab)
        k += 1

    return hist
