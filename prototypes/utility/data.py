import pandas as pd
import os
import cv2
import json
import numpy as np


class DataLoader:
    def __init__(self, data_path, metadata_path):
        self.data_path = data_path
        self.metadata_df = pd.read_csv(metadata_path, engine='python')

    def get_data(self, target, width, height, mode = "color", n_sample=None):
        if n_sample is not None:
            images_path = self.metadata_df.query(f"target=={target}").sample(n=n_sample)["isic_id"].values
        else:
            images_path = self.metadata_df.query(f"target=={target}")["isic_id"].values

        if mode == "color":
            images = np.zeros((len(images_path), width, height, 3), dtype=np.uint8)
        else:
            images = np.zeros((len(images_path), width, height), dtype=np.uint8)

        for i in range(len(images_path)):
                images[i] = cv2.resize(cv2.imread(os.path.join(self.data_path, images_path[i] + ".jpg"),
                                                  cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE),
                                       (width, height))

        return images


class ProjectConfiguration:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def get_keys(self):
        return self.config.keys()

    def get_value(self, key):
        return self.config[key]
