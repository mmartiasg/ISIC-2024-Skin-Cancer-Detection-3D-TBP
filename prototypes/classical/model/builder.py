from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def build_models(parameters=None, dataset=None, feature_set=None):
    if parameters is None:
        return [
            ("Random Forest", RandomForestClassifier()),
            ("Ada Boost", AdaBoostClassifier()),
            ("SVM", SVC()),
            (
                "MLP",
                MLPClassifier(),
            ),
            ("Logistic Regression", LogisticRegression()),
            ("LDA", LinearDiscriminantAnalysis()),
            ("KNN", KNeighborsClassifier()),
            ("GaussianNB", GaussianNB()),
        ]
    return [
        (
            "Random Forest",
            RandomForestClassifier(**parameters[dataset][feature_set]["Random Forest"]),
        ),
        (
            "Ada Boost",
            AdaBoostClassifier(**parameters[dataset][feature_set]["Ada Boost"]),
        ),
        ("SVM", SVC(**parameters[dataset][feature_set]["SVM"])),
        (
            "MLP",
            MLPClassifier(**parameters[dataset][feature_set]["MLP"]),
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                **parameters[dataset][feature_set]["Logistic Regression"]
            ),
        ),
        ("LDA", LinearDiscriminantAnalysis(**parameters[dataset][feature_set]["LDA"])),
        ("KNN", KNeighborsClassifier(**parameters[dataset][feature_set]["KNN"])),
        ("GaussianNB", GaussianNB(**parameters[dataset][feature_set]["GaussianNB"])),
    ]
