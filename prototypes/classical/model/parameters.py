import numpy as np


search_parameters = {
    # "SVM": {
    #     "kernel": ("linear", "rbf", "poly", "sigmoid", "precomputed"),
    #     "C": [2**i for i in range(3)],
    #     "degree": [2**i for i in range(2)],
    #     "gamma": [1e-3, 1e-4],
    #     # "class_weight": [None, "balanced"],
    #     # Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.
    #     # "decision_function_shape": ["ovo", "ovr"],
    # },
    "Random Forest": {
        "n_estimators": [2**i for i in range(8)],
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": [2 ** (i + 1) for i in range(3)],
    },
    "Ada Boost": {
        "n_estimators": [2**i for i in range(8)],
        "algorithm": ["SAMME"],
    },
    "MLP": {
        "activation": ["relu", "tanh", "logistic"],
        "solver": ["adam"],
        "hidden_layer_sizes": [
            (128,),
            (256,),
            (32, 64, 128),
        ],
        "learning_rate": ["adaptive", "invscaling"],
        "learning_rate_init": [1e-2, 1e-3],
        "max_iter": [3000],
        "early_stopping": [True],
    },
    "Logistic Regression": {"C": np.logspace(-4, 4, 8), "max_iter": [3000]},
    "LDA": {"n_components": [1]},
    "KNN": {
        "n_neighbors": [i + 1 for i in range(8)],
        "leaf_size": [10, 20, 30, 50],
        "p": [1, 2],
        "weights": ["uniform", "distance"],
    },
    "GaussianNB": {"var_smoothing": np.linspace(1e-5, 1e-10, 5)},
}
