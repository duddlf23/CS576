import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats: an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels: an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats: an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type: SVM kernel type. 'linear' or 'RBF'

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)
    N = train_image_feats.shape[0]
    M = test_image_feats.shape[0]

    c_n  =len(categories)

    test_y = np.zeros((M, c_n))
    if kernel_type == 'RBF':
        kernel_type = 'rbf'

    # Create 1 vs other svm detectors for the number of categories.
    for k in range(c_n):

        # '1' category to be label 1, 'other' category to be label -1
        train_y = np.repeat(-1, N)
        train_y[train_labels == categories[k]] = 1

        # I use tuned paremater, C = 10, gamma = 'scale' and kernel_type
        model = svm.SVC(C=10, kernel = kernel_type, gamma = 'scale').fit(train_image_feats, train_y)

        # I can get score of test_image_feature of each svm detector
        test_y[:,k] = np.reshape(model.decision_function(test_image_feats), -1)

    # For choose label, I adopt svm detector with the most positive score.
    test_lab = np.argmax(test_y, axis=1)
    test_label = []
    for i in range(M):
        test_label.append(categories[test_lab[i]])

    return np.array(test_label)