import pandas as pd

from prototypes.classical.model.trainner import find_best_hyper_parameters, calculate_metrics_k_folds
from prototypes.classical.model.parameters import search_parameters
import numpy as np
import json
import os, glob
import re
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
import math
import logging


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='training.log', level=logging.INFO)

    with open("config.json", "r") as f:
        config = json.load(f)

    VECTORS_PATH = config["VECTORS_PATH"]
    K_FOLDS = int(config["K_FOLDS"])
    SAMPLE_PERCENTAGE = float(config["SAMPLE_PERCENTAGE"])
    HYPER_PARAMETERS_PATH = config["HYPER_PARAMETERS_PATH"] + "_" + config["VERSION"]

    assert SAMPLE_PERCENTAGE > 0.0 and SAMPLE_PERCENTAGE < 1.0

    features_batch = np.sort(glob.glob(os.path.join(VECTORS_PATH, "gabor_attention_maps", f"{config['IMAGE_WIDTH']}_{config['IMAGE_WIDTH']}", "feature_vector_*.npy")))
    np.random.shuffle(features_batch)

    if not os.path.exists(HYPER_PARAMETERS_PATH):
        find_hyperparameters(K_FOLDS, VECTORS_PATH, config, features_batch, label_balance, logger)

    SPLITS = 4
    TOTAL_SUB_BATCHES = len(features_batch) // SPLITS

    best_hyperparameters = None
    with open(HYPER_PARAMETERS_PATH, "r") as f:
        best_hyper_parameters = json.load(f)
        logger.info("Hyper parameters found!")

    evaluate_algorithms(K_FOLDS, SPLITS, TOTAL_SUB_BATCHES, VECTORS_PATH, config, features_batch, best_hyper_parameters, logger)
    # Train the best algorithm with the best hyperparameters!
    # This might include to sample the dataset to get the most samples possible.
    # I could build a pipeline to read the Hd5 file here or use the generated files and export a model.
    # Then load the model, make a pipeline with the preprocess needed and pickle everything togheter again.


def label_balance(y, logger):
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class balance: {dict(zip(unique, counts))}")


def evaluate_algorithms(K_FOLDS, SPLITS, TOTAL_SUB_BATCHES, VECTORS_PATH, config, features_batch, best_hyperparameters, logger):
    algorithms_metrics = []

    for split_index in range(SPLITS):
        # Train and evaluate each model in each split
        x_features = []
        y_labels = []
        for sample_index in range(TOTAL_SUB_BATCHES):
            x = features_batch[(split_index + 1) * sample_index]
            batch_index = re.findall(r'\d+', x.split("/")[-1])[0]
            y = os.path.join(VECTORS_PATH, "gabor_attention_maps", f"{config['IMAGE_WIDTH']}_{config['IMAGE_WIDTH']}",
                             f"label_{batch_index}.npy")

            x_features.append(np.load(x))
            y_labels.append(np.load(y))

        x_features = np.vstack(x_features).astype(np.float32)
        y_labels = np.vstack(y_labels).astype(np.float32)

        smote = SMOTE(sampling_strategy="auto")
        image_resampled, labels_resampled = smote.fit_resample(x_features, y_labels)
        label_balance(labels_resampled, logger)

        metrics = calculate_metrics_k_folds(x=image_resampled,
                                            y=labels_resampled,
                                            k=K_FOLDS,
                                            dataset_name="train_images",
                                            feature_set="gabor_attention_maps",
                                            parameters=best_hyperparameters,
                                            logger=logger)

        metrics["split"] = split_index
        algorithms_metrics.append(metrics)

        logger.info(f"Metrics for split: {split_index}\n")
        logger.info(metrics)
        logger.info("---------------------------------------\n")
    pd.DataFrame(algorithms_metrics).to_csv("algorithms_metrics.csv", index=False)


def find_hyperparameters(K_FOLDS, VECTORS_PATH, config, features_batch, label_balance, logger):
    N_samples = math.ceil(features_batch.shape[0] * 0.25)
    sampling_index = np.random.choice(features_batch.shape[0], N_samples, replace=False)
    samples = features_batch[sampling_index]
    # Find best hyper-parameters for all algorithms
    print(f"Find best Hyper-parameters on {samples.shape[0] * int(config['BATCH_SIZE'])} samples")
    x_features = []
    y_labels = []
    for sample_index in range(N_samples):
        x = features_batch[sample_index]
        batch_index = re.findall(r'\d+', x.split("/")[-1])[0]
        y = os.path.join(VECTORS_PATH, "gabor_attention_maps", f"{config['IMAGE_WIDTH']}_{config['IMAGE_WIDTH']}",
                         f"label_{batch_index}.npy")

        x_features.append(np.load(x))
        y_labels.append(np.load(y))
    x_features = np.vstack(x_features).astype(np.float32)
    y_labels = np.vstack(y_labels).astype(np.float32)
    logger.info("Before SMOTE: y count")
    label_balance(y=y_labels, logger=logger)
    smote = SMOTE(sampling_strategy="minority")
    image_resampled, labels_resampled = smote.fit_resample(x_features, y_labels)
    logger.info("After SMOTE: y count")
    label_balance(y=labels_resampled, logger=logger)
    training_dict = {}
    training_dict["train_images"] = {}
    training_dict["train_images"]["x"] = {"gabor_attention_maps": image_resampled}
    training_dict["train_images"]["y"] = labels_resampled
    best_hyper_parameters = find_best_hyper_parameters(training_data=training_dict,
                                                       parameters=search_parameters,
                                                       k_folds=K_FOLDS)
    logger.info("Best Hyperparameters")
    logger.info(best_hyper_parameters)

    with open("best_hyperparameters_0.1.0", "w") as f:
        f.write(json.dumps(dict(best_hyper_parameters)))


if __name__ == "__main__":
    main()