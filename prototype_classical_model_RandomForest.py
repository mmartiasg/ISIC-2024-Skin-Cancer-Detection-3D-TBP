# warm_start – When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution. See the Glossary
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import json
import logging
from prototypes.deeplearning.dataloader.IsicDataLoader import LoadDataVectors, LoadPreProcessVectors
import torch
import os
from prototypes.classical.descriptors.vetorizer import GaborAttentionLBPVectors
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from sklearn import pipeline
from joblib import dump, load
import pandas as pd


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='training_prototype_model.log', level=logging.INFO)

    with open("config.json", "r") as f:
        config = json.load(f)

    DATASET_PATH = config["DATASET_PATH"]
    MODEL_VERSION = config["VERSION"]
    HYPER_PARAMETERS_PATH = config["HYPER_PARAMETERS_PATH"] + "_" + MODEL_VERSION

    # with open(HYPER_PARAMETERS_PATH, "r") as f:
    #     best_hyper_parameters = json.load(f)
    #     logger.info("Hyper parameters found!")

    best_hyper_parameters = {
                                "criterion":'log_loss',
                                "min_samples_split":8,
                                "n_estimators":16
                        }

    # model = RandomForestClassifier(**best_hyper_parameters["train_images"]["gabor_attention_maps"]["Random Forest"])
    model = RandomForestClassifier(**best_hyper_parameters)

    x = []
    y = []

    preprocess_vectors = LoadPreProcessVectors(dataset_base_path="feature_vectors",
                                               feature_name="gabor_attention_maps", target_index=[0], dimensions=128)

    vector_dataloader = torch.utils.data.DataLoader(preprocess_vectors, batch_size=8, shuffle=True, num_workers=1)

    for batch in vector_dataloader:
        x.append(batch[0].numpy())
        y.append(batch[1].numpy())
        break

    x = np.vstack(np.vstack(x, dtype=np.float32))
    y = np.vstack(np.vstack(y, dtype=np.float32))

    # need big batch to have several samples from the minor class
    smote = SMOTE(sampling_strategy="auto")
    x, y = smote.fit_resample(x, y)

    model.fit(x, y)

    dump(model, f"random_forest_{MODEL_VERSION}.joblib")
    exported_model = load(f"random_forest_{MODEL_VERSION}.joblib")

    load_vectors_test = LoadDataVectors(hd5_file_path=os.path.join(DATASET_PATH, "test-image.hdf5"),
                                        transform=GaborAttentionLBPVectors())

    vector_dataloader_test = torch.utils.data.DataLoader(load_vectors_test, batch_size=8, num_workers=1)

    predictions = []

    logger.info("predictions model...")
    for element in vector_dataloader_test:
        x, key = element
        preds = exported_model.predict(x.numpy())

        predictions.extend(zip(key, preds))

    submit = pd.DataFrame(predictions, columns=["isic_id", "target"])

    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()