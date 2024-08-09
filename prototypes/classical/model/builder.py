from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB


def build_models(parameters=None, dataset=None, feature_set=None):
    if parameters is None:
        return [
            ("GradientBoostingClassifier", GradientBoostingClassifier()),
            ("HistGradientBoostingClassifier", HistGradientBoostingClassifier()),
            ("Perceptron", Perceptron()),
            ("SGDClassifier", SGDClassifier()),
            ("PassiveAggressiveClassifier", PassiveAggressiveClassifier()),
            ("BernoulliNB", BernoulliNB()),
            ("MultinomialNB", MultinomialNB()),
            ("GaussianNB", GaussianNB()),
        ]
    return [
        (
            "GradientBoostingClassifier", GradientBoostingClassifier(**parameters[dataset][feature_set]["GradientBoostingClassifier"])
        ),
        (
            "HistGradientBoostingClassifier", HistGradientBoostingClassifier(**parameters[dataset][feature_set]["HistGradientBoostingClassifier"])
        ),
        (
            "Perceptron", Perceptron(**parameters[dataset][feature_set]["Perceptron"])
        ),
        (
            "SGDClassifier", SGDClassifier(**parameters[dataset][feature_set]["SGDClassifier"])
        ),
        (
            "PassiveAggressiveClassifier", PassiveAggressiveClassifier(**parameters[dataset][feature_set]["PassiveAggressiveClassifier"])
        ),
        (
            "BernoulliNB", BernoulliNB(**parameters[dataset][feature_set]["BernoulliNB"])
        ),
        (
            "MultinomialNB", MultinomialNB(**parameters[dataset][feature_set]["MultinomialNB"])
        ),
        (
            "GaussianNB", GaussianNB(**parameters[dataset][feature_set]["GaussianNB"])
        )
    ]
