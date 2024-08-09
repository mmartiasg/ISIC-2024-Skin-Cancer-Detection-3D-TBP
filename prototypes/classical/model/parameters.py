import numpy as np


search_parameters = {
    "GradientBoostingClassifier": {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.01, 0.05, 0.2],
        'max_depth': [2, 4, 8, None],
        'min_samples_split': [2, 4, 8, 16],
        'min_samples_leaf': [2, 4, 8, 16],
        'max_features': ['sqrt', 'log2'],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    },
    "HistGradientBoostingClassifier": {
        'learning_rate': [0.001, 0.01, 0.05, 0.2],
        'max_iter': [100, 200, 300, 400],
        'max_leaf_nodes': [15, 31, 63, 127],
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'max_bins': [255, 512],
        'l2_regularization': [0.0, 0.1, 0.5, 1.0],
        'early_stopping': [True, False],
        'scoring': ['loss', 'accuracy']  # only if early_stopping is True
    },
    "SGDClassifier": {
        "penalty": ["l2", "l1"],
        "fit_intercept": [True, False],
        "max_iter": [1000, 2000, 3000],
        "alpha": np.linspace(0, 0.3,4),
        "eta0": np.linspace(0, 1.0,4),
        "n_jobs": -1,
        "learning_rate": ['invscaling', 'constant', 'adaptive', 'optimal'],
        "power_t": np.linspace(0.25, 1.0,4),
    },
    "PassiveAggressiveClassifier": {
        "C": np.linspace(0, 1, 4),
        "fit_intercept": [True, False],
        "n_jobs": -1,
        "max_iter": [1000, 2000, 2500],
        "loss": ["squared_hinge", "hinge"]
    },
    "GaussianNB": {
        "var_smoothing": np.linspace(1e-5, 1e-10, 5)},
    "BernoulliNB": {
        "alpha": np.linspace(0, 3.0, 4),
        "binarize": [True, False],
        "fit_prior": [True, False]
    },
    "Perceptron":{
        "penalty": ["l2", "l1"],
        "fit_intercept": [True, False],
        "max_iter": [1000, 2000, 2500],
        "eta0": np.linspace(0, 1, 4),
        "alpha": np.linspace(0, 3.0, 4),
        "n_jobs": -1
    }
}
