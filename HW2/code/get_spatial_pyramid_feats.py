import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_hog.npy' (for HoG) or 'vocab_sift.npy' (for SIFT)
    exists and contains an N x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]

    # number of kind of histogram
    n2 = int((4 ** (max_level + 1) - 1) / 3)

    N = len(image_paths)

    # feature dimension is vocab_size *  (1 / 3) * (4 ^ (max_level + 1) - 1)
    hist = np.zeros((N, vocab_size * n2))

    # Your code here. You should also change the return value.
    k = 0
    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]
        n = img.shape[0]
        m = img.shape[1]

        id = 0
        for l in range(max_level + 1):

            # I choose the scale 1/4 when level 0, 1/4 when level 1, 1/2 when level 2, as in the paper.
            scale = 2 ** (l - max_level - 1)
            if l == 0:
                scale = 2 ** -2

            # split image to 2^l * 2^l subimages, and get feature and histogram each splitted images.
            div = 2 ** l
            dx = int(n / div)
            dy = int(m / div)
            for i in range(div):
                for j in range(div):
                    x = i * dx
                    y = j * dy
                    features = feature_extraction(img[x:x+dx, y:y+dy], feature)
                    if features is not None:
                        d = pdist(features, vocab);
                        lab = np.argmin(d, axis=1)
                        for l in lab:
                            hist[k, id + l] += 1 * scale
                    id += vocab_size

        # normalizing
        hist[k,:] = hist[k,:] / np.sum(hist[k,:])
        k += 1

    return hist


