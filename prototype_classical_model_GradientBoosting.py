# warm_start – When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution. See the Glossary
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
import json
import logging
from prototypes.deeplearning.dataloader.IsicDataLoader import LoadDataVectors, LoadPreProcessVectors
import torch
import os
from prototypes.classical.descriptors.vetorizer import GaborAttentionLBPVectors
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from tqdm.auto import tqdm
from joblib import dump, load
import pandas as pd


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='training_prototype_model.log', level=logging.INFO)

    with open("config.json", "r") as f:
        config = json.load(f)

    DATASET_PATH = config["DATASET_PATH"]
    MODEL_VERSION = config["VERSION"]

    # TODO: unccoment when fixed
    HYPER_PARAMETERS_PATH = config["HYPER_PARAMETERS_PATH"] + "_" + MODEL_VERSION
    with open(HYPER_PARAMETERS_PATH, "r") as f:
        best_hyper_parameters = json.load(f)
        logger.info("Hyper parameters found!")

    # best_hyper_parameters = {
    #     'n_estimators': 300,
    #     'learning_rate': 0.001,
    #     'max_depth': 4,
    #     'min_samples_split': 8,
    #     'min_samples_leaf': 8,
    #     'max_features': 'sqrt',
    #     'subsample': 0.9,
    #     'warm_start': True
    # }

    # to train using batches...
    best_hyper_parameters["train_images"]["gabor_attention_maps"]["GradientBoostingClassifier"]["warm_start"] = True

    model = GradientBoostingClassifier(**best_hyper_parameters["train_images"]["gabor_attention_maps"]["GradientBoostingClassifier"])
    # model = GradientBoostingClassifier(**best_hyper_parameters)

    # load_vectors = LoadDataVectors(hd5_file_path=os.path.join(DATASET_PATH, "train-image.hdf5"),
    #                                metadata_csv_path=os.path.join(DATASET_PATH, "train-metadata.csv"),
    #                                target_columns=["target"],
    #                                transform=GaborAttentionLBPVectors())
    # vector_dataloader = torch.utils.data.DataLoader(load_vectors, batch_size=16000, shuffle=True, num_workers=8)

    preprocess_vectors = LoadPreProcessVectors(dataset_base_path="feature_vectors",
                                               feature_name="gabor_attention_maps", target_index=[0], dimensions=128)

    vector_dataloader = torch.utils.data.DataLoader(preprocess_vectors, batch_size=12, shuffle=True, num_workers=4)

    for batch in tqdm(vector_dataloader, total=len(vector_dataloader)):
        logger.info("loading batch...")

        #this returns a batch of batches
        x = np.vstack(batch[0].numpy())
        y = np.vstack(batch[1].numpy())

        # need big batch to have several samples from the minor class
        logger.info("Oversamping....")
        smote = SMOTE(sampling_strategy="auto")
        x, y = smote.fit_resample(x, y)
        logger.info("fiting batch....")
        model.fit(x, y)

    logger.info("saving model...")
    dump(model, f"gradient_boosting_classic_{MODEL_VERSION}.joblib")
    logger.info("recovering model...")
    exported_model = load(f"gradient_boosting_classic_{MODEL_VERSION}.joblib")

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