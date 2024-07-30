import pandas as pd
import cv2 as cv
import os


class IsiCancerData:
    def __init__(self, config_file):
        self.config_file = config_file
        self.train = pd.read_csv(config_file["TRAIN_METADATA"]).to_numpy()
        self.index = 0
        self.length = self.train.shape[0]
        self.train_images_path = config_file["TRAIN_IMAGES_PATH"]

    def get_item(self):
        if self.index > self.length:
            return None

        image = cv.imread(os.join(self.train_images_path[self.index]), self.train[self.index][0])
        labels = self.train[self.index][1:]
        self.index += 1

        return image, labels
