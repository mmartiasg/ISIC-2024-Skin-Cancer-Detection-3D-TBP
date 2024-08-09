import numpy as np
import sklearn
from prototypes.classical.descriptors.texture import LBPTransformer, GaborTransformer


class LBPVectorizer(sklearn.base.TransformerMixin):
    def __init__(self):
        super(LBPVectorizer, self).__init__()

    def transform(self, X):
        count, bin = np.histogram(X, bins=255)

        return count/count.sum()


class HistogramVectorizer(sklearn.base.TransformerMixin):
    def __init__(self):
        super(HistogramVectorizer, self).__init__()

    def transform(self, X):
        transformed = np.zeros((len(X), 254))

        for i in range(len(X)):
            transformed[i] = np.histogram(X[i], bins=range(255))[0]

        return transformed


class GaborAttentionLBPVectors(sklearn.base.TransformerMixin):
    def __init__(self):
        super(GaborAttentionLBPVectors, self).__init__()
        self.lbp_transformer = LBPTransformer(p=8, r=1, method="ror")
        self.lbp_vectorizer = LBPVectorizer()
        self.gabor_banks = []
        for theta in [np.pi, np.pi / 2, np.pi / 4]:
            self.gabor_banks.append(GaborTransformer(frequency=1 / 100, theta=theta, sigma_x=5, sigma_y=5))

    def transform(self, X):
        feature_vector_bank = np.zeros((len(self.gabor_banks), 255 * 3))

        for bank_index, gabor_transformer in enumerate(self.gabor_banks):
            x_imag = gabor_transformer.transform(X)[1]
            attention_map = X.copy()

            attention_map[:, :, 0] = attention_map[:, :, 0] * (x_imag > 0)
            attention_map[:, :, 1] = attention_map[:, :, 1] * (x_imag > 0)
            attention_map[:, :, 2] = attention_map[:, :, 2] * (x_imag > 0)

            lbp_map_channel_1 = self.lbp_transformer.transform(attention_map[:, :, 0])
            lbp_map_channel_2 = self.lbp_transformer.transform(attention_map[:, :, 1])
            lbp_map_channel_3 = self.lbp_transformer.transform(attention_map[:, :, 2])

            feature_vector_bank[bank_index] = np.hstack((self.lbp_vectorizer.transform(lbp_map_channel_1),
                                                         self.lbp_vectorizer.transform(lbp_map_channel_2),
                                                         self.lbp_vectorizer.transform(lbp_map_channel_3)))

        return np.hstack(feature_vector_bank)
