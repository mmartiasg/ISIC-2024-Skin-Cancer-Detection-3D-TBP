import numpy as np
import sklearn
from prototypes.classical.descriptors.texture import LBPTransformer, GaborTransformer


class LBPVectorizer(sklearn.base.TransformerMixin):
    def __init__(self):
        super(LBPVectorizer, self).__init__()

    def transform(self, X):
        count, bin = np.histogram(X, bins=255)

        return count/count.sum()
