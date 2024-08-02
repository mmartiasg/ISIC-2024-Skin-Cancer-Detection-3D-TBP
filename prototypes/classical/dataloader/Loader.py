import pandas as pd
import cv2 as cv
import os
import numpy as np


class IsiCancerData:
    def __init__(self, config_file):
        self.config_file = config_file
        self.train = pd.read_csv(config_file["TRAIN_METADATA"], engine="python")[config_file["TARGET_COLUMNS"].split(",")].to_numpy()
        self.index = 0
        self.length = self.train.shape[0]
        self.train_images_path = config_file["TRAIN_IMAGES_PATH"]
        self.image_width = config_file["IMAGE_WIDTH"]
        self.image_height = config_file["IMAGE_HEIGHT"]
        self.batch_size = config_file["BATCH_SIZE"]

    def get_item(self):
        if self.index >= self.length:
            return None

        # first one is the image id.
        image = cv.imread(os.path.join(self.train_images_path, self.train[self.index][0]+".jpg"))
        resized_image = cv.resize(
            image, (self.image_height, self.image_width), interpolation=cv.INTER_CUBIC
        )
        labels = self.train[self.index][1:]
        self.index += 1

        return resized_image, labels

    def reset_index(self):
        self.index = 0

    def get_next_batch(self):
        np.random.shuffle(self.train)

        for i in range(self.batch_size, self.length, self.batch_size):

            # first one is the image id.
            images = np.zeros((self.batch_size, self.image_height, self.image_width, 3), dtype=np.uint8)

            for e in range(self.batch_size):
                if i + e >= self.length:
                    print("break")
                    break

                image = cv.imread(os.path.join(self.train_images_path, self.train[i+e][0] + ".jpg"))
                image = cv.resize(
                    image, (self.image_height, self.image_width), interpolation=cv.INTER_CUBIC
                )
                images[e] = image

            labels = self.train[i: i+self.batch_size, 1:]

            yield images, labels


    def get_next_batch_balanced(self, strategy="minority"):
        smote = SMOTE(sampling_strategy=strategy)
        np.random.shuffle(self.train)

        for i in range(self.batch_size, self.length, self.batch_size):

            # first one is the image id.
            images = np.zeros((self.batch_size, self.image_height, self.image_width, 3), dtype=np.uint8)

            for e in range(self.batch_size):
                if i + e >= self.length:
                    print("break")
                    break

                image = cv.imread(os.path.join(self.train_images_path, self.train[i+e][0] + ".jpg"))
                image = cv.resize(
                    image, (self.image_height, self.image_width), interpolation=cv.INTER_CUBIC
                )
                images[e] = image

            labels = self.train[i: i+self.batch_size, 1:].astype(np.float16)

            image_resampled, labels_resampled = smote.fit_resample(images.reshape(-1, images.shape[1] * images.shape[2]), labels)

            yield image_resampled, labels_resampled


    def total_samples(self):
        return (self.length//self.batch_size)
