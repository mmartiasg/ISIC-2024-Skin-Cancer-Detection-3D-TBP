from prototypes.classical.model.trainner import find_best_hyper_parameters, calculate_metrics_k_folds
from prototypes.classical.model.parameters import search_parameters
import numpy as np
import json
import os, glob
import re
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE


with open("config.json", "r") as f:
    config = json.load(f)

smote = SMOTE(sampling_strategy="minority")

VECTORS_PATH = config["VECTORS_PATH"]

features_batch = np.sort(glob.glob(os.path.join(VECTORS_PATH, "gabor_attention_maps", f"{config['IMAGE_WIDTH']}_{config['IMAGE_WIDTH']}", "feature_vector_*.npy")))
np.random.shuffle(features_batch)

x_features = []
y_labels = []

TOTAL_BATCHES = len(features_batch)
SPLITS = 5
TOTAL_SUB_BATCHES = len(features_batch) // SPLITS

for split_index in range(SPLITS):
    for sample_index in range(TOTAL_SUB_BATCHES):
        x = features_batch[(split_index + 1) * sample_index]
        batch_index = re.findall(r'\d+', x.split("/")[-1])[0]
        y = os.path.join(VECTORS_PATH, "gabor_attention_maps", f"{config['IMAGE_WIDTH']}_{config['IMAGE_WIDTH']}", f"label_{batch_index}.npy")

        x_features.append(np.load(x))
        y_labels.append(np.load(y))

    x_features = np.vstack(x_features).astype(np.float32)
    y_labels = np.vstack(y_labels).astype(np.float32)

    image_resampled, labels_resampled = smote.fit_resample(x_features, y_labels)

    training_dict = {}
    training_dict["train_images"] = {}
    training_dict["train_images"]["x"] = {"gabor_attention_maps" : np.array(image_resampled)}
    training_dict["train_images"]["y"] = labels_resampled

    best_hyper_parameters = find_best_hyper_parameters(training_data=training_dict,
                                                       parameters=search_parameters,
                                                       k_folds=10)

    print(best_hyper_parameters)
    print("---------------------------------------")