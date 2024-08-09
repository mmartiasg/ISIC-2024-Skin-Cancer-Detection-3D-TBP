import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
import sklearn
import cv2
from scipy import ndimage as ndi


class BlackBarsRemover(sklearn.base.TransformerMixin):
    def __init__(self):
        super(BlackBarsRemover, self).__init__()

    def remove_bars(self, image):
        min_width = min_height = 0
        image_without_black_bars = image.copy()

        for i in range(image.shape[0]):
            if image[i, :].sum() < image.shape[0]:
                min_height = i + 1

        for j in range(image.shape[1]):
            if image[:, j].sum() < image.shape[1]:
                min_width = j

        return cv2.resize(image_without_black_bars[min_height:, min_width:],
                          (image.shape[0], image.shape[1]),
                          interpolation=cv2.INTER_CUBIC)

    def transform(self, X):
        transformed_array = np.zeros_like(X)

        for i in range(len(X)):
            transformed_array[i, :] = self.remove_bars(X[i])

        return transformed_array


class OtsuThresholdingSegmentation(sklearn.base.TransformerMixin):
    def __init__(self):
        super(OtsuThresholdingSegmentation, self).__init__()

    def otsu_threshold(self, image):
        image_blured = cv2.GaussianBlur(image.copy(), (5, 5), 0)
        otsu_threshold, image_result = cv2.threshold(
            image_blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return otsu_threshold

    def transform(self, X):
        transformed_array = np.zeros_like(X)

        for i in range(len(X)):
            transformed_array[i, :] = X[i] * (X[i] < self.otsu_threshold(X[i]))

        return transformed_array
