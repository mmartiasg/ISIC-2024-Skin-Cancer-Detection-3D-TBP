import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
import sklearn
import cv2 as cv
from scipy import ndimage as ndi


class LBPTransformer(sklearn.base.TransformerMixin):
    def __init__(self, p, r, method="ror"):
        self.p = p
        self.r = r
        self.method = method

    def transform(self, X):
        gray_scale = cv.cvtColor(X, cv.COLOR_BGR2GRAY) if X.ndim == 3 else X

        return local_binary_pattern(image=gray_scale, P=self.p, R=self.r, method=self.method)


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

    def transform(self, X):
        gray_scale = cv.cvtColor(X, cv.COLOR_BGR2GRAY) if X.ndim == 3 else X

        kernel = gabor_kernel(frequency=self.frequency,
                              theta=self.theta,
                              sigma_x=self.sigma_x,
                              sigma_y=self.sigma_y)

        if self.just_imag:
            return ndi.convolve(gray_scale, np.imag(kernel), mode='reflect', cval=0)

        return ndi.convolve(gray_scale, np.real(kernel), mode='reflect', cval=0), ndi.convolve(gray_scale, np.imag(kernel), mode='reflect', cval=0)
