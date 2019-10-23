import numpy as np


def get_features_from_pca(feat_num, feature):
    """
    This function loads 'vocab_sift.npy' or 'vocab_hog.npg' file and
    returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
    :param feature: 'Hog' or 'SIFT'

    :return: an N x feat_num matrix
    """

    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    # Your code here. You should also change the return value.
    n = vocab.shape[0]
    f_n = vocab.shape[1]
    f = np.copy(vocab)

    # subtract average of each coordinate
    for i in range(f_n):
        f[:,i] = f[:,i] - np.average(f[:,i])

    # make covariance matrix using matrix multiplication (X - X_bar)^T * (X - X_bar) / N
    cov = np.matmul(np.transpose(f), f)
    cov = cov / n

    # The eigenvector corresponding to dominate eigenvalue is equal to the first,second,.. column vector of V when a singular value decomposition.
    u, s, vh = np.linalg.svd(cov, full_matrices=True)
    v = np.transpose(vh)

    # In order to obtain the principal component of each voca, I compute the inner product with eigenvector.
    pca_f = np.zeros((n, feat_num))
    for i in range(feat_num):
        pca_f[:,i] = np.reshape(np.matmul(f, np.reshape(v[:,i], (-1,1))), -1)

    return pca_f


