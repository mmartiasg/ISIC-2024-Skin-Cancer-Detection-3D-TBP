from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    homogeneity_completeness_v_measure,
    pair_confusion_matrix,
    adjusted_rand_score,
)
from sklearn.utils import shuffle
import pandas as pd
from prototypes.classical.model.builder import build_models
import multiprocessing as mpt
from sklearn import preprocessing


def k_fold_model(k, x, y, model, metrics_function, metrics_names, logger):

    evaluated_metrics_fold = []

    sfk = StratifiedKFold(n_splits=k, shuffle=True)
    sfk.get_n_splits(x, y)

    for i, (train_index, val_index) in tqdm(enumerate(sfk.split(x, y))):

        train_data_segment = x[train_index]
        train_label_segment = y[train_index]

        val_data_segment = x[val_index]
        val_label_segment = y[val_index]

        copy_model = clone(model)

        copy_model.fit(train_data_segment, train_label_segment)

        y_pred = copy_model.predict(val_data_segment)

        metrics = []
        for m_function in metrics_function:
            if len(np.unique(y)) > 1:
                if cohen_kappa_score == m_function:
                    metrics.append(m_function(val_label_segment, y_pred, weights=None))
                else:
                    metrics.append(
                        m_function(val_label_segment, y_pred, average="weighted")
                    )

        metric_dict = dict(zip(metrics_names, metrics))
        metric_dict["cv_fold"] = i
        evaluated_metrics_fold.append(metric_dict)

        if logger is not None:
            logger.info(f"model: {model} | metrics: {metric_dict} | fold: {i}")

    return evaluated_metrics_fold


def calculate_metrics_k_folds(x, y, dataset_name, feature_set, k=4, parameters=None, logger=None):
    metrics = []

    x_shuffle, y_shuffle = shuffle(x, y)
    # x_shuffle = preprocessing.scale(x_shuffle)

    for model_name, model in tqdm(build_models(parameters, dataset_name, feature_set)):
        metric_results = k_fold_model(
            k,
            x_shuffle,
            y_shuffle,
            model,
            [f1_score, cohen_kappa_score, precision_score, recall_score],
            [
                "f1_score",
                "cohen_kappa_score",
                "precision_score",
                "recall_score",
            ],
            logger=logger
        )

        for metric_result in metric_results:
            metric_result["algorithm"] = model_name
            metric_result["dataset"] = dataset_name
            metric_result["feature_set"] = feature_set

            metrics.append(metric_result)

    return metrics


def find_best_hyper_parameters(training_data, parameters, k_folds):
    best_parameters_per_model = {}
    scorer = make_scorer(f1_score)

    for dataset_name in tqdm(training_data.keys()):
        best_parameters_per_model[dataset_name] = {}
        for feature_set in training_data[dataset_name]["x"].keys():

            x = training_data[dataset_name]["x"][feature_set]
            # x = preprocessing.scale(x)

            y = training_data[dataset_name]["y"]
            best_parameters_per_model[dataset_name][feature_set] = {}

            for model in build_models():
                name = model[0]
                model_instance = model[1]
                print(
                    f"Model: {name} | dataset: {dataset_name} | feature set: {feature_set}"
                )

                parameter_space_search = parameters[name]
                grid_search = GridSearchCV(
                    estimator=model_instance,
                    param_grid=parameter_space_search,
                    n_jobs=(mpt.cpu_count() - 2),
                    cv=k_folds,
                    verbose=3,
                    scoring=scorer,
                )
                try:
                    grid_search.fit(X=x, y=y)
                    best_parameters_per_model[dataset_name][feature_set][
                        name
                    ] = grid_search.best_params_
                except:
                    print(f"Could not fit the model: {model}")

    return best_parameters_per_model


def get_evaluation_results(training_dict, best_hyper_parameters, k_folds=10):
    results = []

    for dataset_name in training_dict.keys():
        for feature_set in training_dict[dataset_name]["x"].keys():
            metrics = calculate_metrics_k_folds(
                x=training_dict[dataset_name]["x"][feature_set],
                y=training_dict[dataset_name]["y"],
                dataset_name=dataset_name,
                feature_set=feature_set,
                k=k_folds,
                parameters=best_hyper_parameters,
            )
            results.append(metrics)

    return pd.concat([pd.DataFrame(row) for row in results])
