import cv2
import numpy as np


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a N x feature_size matrix.
    """

    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64


        # Your code here. You should also change the return value.

        # make HOG descriptor model with given parameter
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)

        m = img.shape[0]
        n = img.shape[1]

        all_f = []

        # divide original image with 16 X 16 grid and make subimages, so find HoG descriptor by grouped 4 cell (32 X 32) and 16 X 16 stride
        for i in range(int(m / 16) - 1):
            for j in range(int(n / 16) - 1):
                x = i * 16
                y = j * 16
                h = hog.compute(img[x:x+32, y:y+32])
                all_f.append(np.reshape(h, (1, 36)))

        # combine Hog desciptor from sub images
        all_f = np.concatenate(all_f, 0)
        return all_f

    elif feature == 'SIFT':

        # Your code here. You should also change the return value.
        m = img.shape[0]
        n = img.shape[1]

        all_f = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        # divide original image with 20 X 20 grid and make subimages, so find SIFT descriptor by 20 X 20 sub images.
        for i in range(int(m / 20)):
            for j in range(int(n / 20)):
                x = i * 20
                y = j * 20
                kp, des = sift.detectAndCompute(gray[x:x+20, y:y+20], None)
                if len(kp) != 0:
                    all_f.append(des)

        #If sift is not detected, exception handling is done.
        if len(all_f) != 0:
            all_f = np.concatenate(all_f, 0)
        else:
            all_f = None

        return all_f




