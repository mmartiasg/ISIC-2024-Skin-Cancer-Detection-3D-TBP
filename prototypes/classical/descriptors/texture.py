import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
import sklearn
import cv2 as cv
from scipy import ndimage as ndi
import joblib
import multiprocessing as mpt
import math


class LBPTransformer(sklearn.base.TransformerMixin):
    def __init__(self, p, r, method="ror"):
        self.p = p
        self.r = r
        self.method = method

    def transform(self, X):
        transformed_data = np.zeros_like(X)
        for i in range(len(X)):
            transformed_data[i, :] = local_binary_pattern(image=X[i], P=self.p, R=self.r, method=self.method)
        return transformed_data


class HoGTransformer(sklearn.base.TransformerMixin):
    def __init__(self, orientations, pixels_per_cell, cells_per_block, visualize=False):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize

    def transform(self, X):
        gray_scale = cv.cvtColor(X, cv.COLOR_BGR2GRAY) if X.ndim == 3 else X

        return hog(gray_scale, orientations=self.orientations,
                   pixels_per_cell=self.pixels_per_cell,
                   cells_per_block=self.cells_per_block,
                   visualize=self.visualize)


class GaborTransformer(sklearn.base.TransformerMixin):
    def __init__(self, frequency, theta, sigma_x, sigma_y, just_imag=False):
        self.frequency = frequency
        self.theta = theta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.just_imag = just_imag
        self.kernel = gabor_kernel(frequency=self.frequency,
                              theta=self.theta,
                              sigma_x=self.sigma_x,
                              sigma_y=self.sigma_y)

    def transform_batch(self, X):
        transformed_data = np.zeros_like(X)

        for i in range(len(X)):
            transformed_data[i] = np.sqrt(ndi.convolve(X[i], np.real(self.kernel), mode='reflect', cval=0)**2 +
                                   ndi.convolve(X[i], np.imag(self.kernel), mode='reflect', cval=0)**2)

        return transformed_data

    def transform(self, X):
        cpu_count = mpt.cpu_count()
        batch_size = math.ceil(len(X) / cpu_count) + 1

        with joblib.Parallel(n_jobs=cpu_count) as parallel:
            results = parallel(joblib.delayed(self.transform_batch)
                               (X[i: i + batch_size]) for i in
                               range(0, len(X), batch_size))
        return np.vstack(results)


class GaborTransformerBank(sklearn.base.TransformerMixin):
    def __init__(self, gabor_banks):
        super(GaborTransformerBank, self).__init__()
        self.gabor_banks = gabor_banks

    def transform(self, X):
        transformed_data = []
        bank_transformed_data = []

        for gabor_filter_index in range(len(self.gabor_banks)):
            bank_transformed_data.append(self.gabor_banks[gabor_filter_index].transform(X))
        transformed_data.append(bank_transformed_data)

        return np.transpose(np.vstack(transformed_data), axes=[1, 0, 2, 3])
