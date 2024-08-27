import cv2
import torch
import h5py
from PIL import Image
import pandas as pd
import io
import numpy as np
import glob
import os
import re
from prototypes.classical.segmentation.transformers import OtsuThresholdingSegmentation, BlackBarsRemover
from sklearn.model_selection import StratifiedKFold
import math
import copy
from tqdm.auto import tqdm
import albumentations as A


def create_folds(isic_id, metadata, labels, config):
    stratified_kf = StratifiedKFold(n_splits=config.get_value("K_FOLDS"), shuffle=True)

    fold_config = {}
    for i in range(config.get_value("K_FOLDS")):
        fold_config[f"{i + 1}-train"] = {}
        fold_config[f"{i + 1}-val"] = {}

    for fold_index, indexes in enumerate(stratified_kf.split(isic_id, labels)):
        train_index, val_index = indexes

        fold_config[f"{fold_index + 1}-train"]["isic_id"] = isic_id[train_index]
        fold_config[f"{fold_index + 1}-train"]["metadata"] = metadata[train_index]
        fold_config[f"{fold_index + 1}-train"]["target"] = labels[train_index]
        fold_config[f"{fold_index + 1}-val"]["isic_id"] = isic_id[val_index]
        fold_config[f"{fold_index + 1}-val"]["metadata"] = metadata[val_index]
        fold_config[f"{fold_index + 1}-val"]["target"] = labels[val_index]

    return fold_config


class AugmentationWrapper():
    def __init__(self, augmentation_transform):
        self.augmentation_transform = augmentation_transform

    def __call__(self, sample):
        return Image.fromarray(self.augmentation_transform(image=sample)["image"])


class IsicDataLoader(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        super(IsicDataLoader, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
        self.transform = target_transform

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        x = self.train_images[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class LoadDataVectors(torch.utils.data.Dataset):
    def __init__(self, hd5_file_path, metadata_csv_path=None, metadata_columns=None, transform=None, split="train", target_transform=None):
        super(LoadDataVectors, self).__init__()

        self.hd5_file = h5py.File(hd5_file_path, "r")
        self.keys = list(self.hd5_file.keys())
        self.preprocess = BlackBarsRemover()
        self.metadata_dataframe = None
        self.transform = None
        self.metadata_inputs = None
        self.split = split
        self.transform = transform

        if metadata_csv_path is not None:
            self.metadata_dataframe = pd.read_csv(metadata_csv_path, engine="python")
            if metadata_columns is not None:
                self.metadata_inputs = self.metadata_dataframe[metadata_columns].values
            self.target_dict = dict(zip(self.metadata_dataframe["isic_id"].values, self.metadata_dataframe["target"].values))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        x = Image.open(io.BytesIO(self.hd5_file[self.keys[idx]][()]))

        if self.metadata_inputs is not None:
            metadata_x = self.metadata_inputs[idx]

        if self.transform:
            x = self.transform(x)

        if self.metadata_inputs is not None:
            return x, torch.tensor(metadata_x), torch.tensor([self.target_dict[self.keys[idx]]])

        if self.metadata_dataframe is not None:
            return x, torch.tensor([self.target_dict[self.keys[idx]]])

        return x, self.keys[idx]


class LoadPreProcessVectors(torch.utils.data.Dataset):
    def __init__(self, dataset_base_path, feature_name, dimensions, target_index=None, transform=None, target_transform=None):
        super(LoadPreProcessVectors, self).__init__()

        self.feature_vectors_path = glob.glob(os.path.join(dataset_base_path, feature_name, f"{dimensions}_{dimensions}", "feature_vector_*.npy"))
        self.dataset_base_path = dataset_base_path
        self.feature_name = feature_name
        self.dimensions = dimensions
        self.target_index = target_index

    def __len__(self):
        return len(self.feature_vectors_path)

    def __getitem__(self, idx):
        x = np.load(self.feature_vectors_path[idx])
        batch_index = re.findall(r'\d+', self.feature_vectors_path[idx].split("/")[-1])[0]

        y = np.load(os.path.join(self.dataset_base_path,
                         self.feature_name,
                         f"{self.dimensions}_{self.dimensions}",
                         f"label_{batch_index}.npy"))

        return x, y


def over_under_sample(anomaly_images, normal_images, config, augmentation_transform, total_samples=10000, imbalance_percentage=0.5):
    assert total_samples > anomaly_images.shape[0] * 2
    assert 0 < imbalance_percentage <= 0.5

    normal_percentage = total_samples * 0.45 / normal_images.shape[0]
    iterations = math.ceil(normal_images.shape[0] * normal_percentage / anomaly_images.shape[0])
    sampled_ids = np.random.choice(normal_images, math.ceil(total_samples * imbalance_percentage))

    augmented_images = []
    for _ in tqdm(range(iterations)):
        for image_name in anomaly_images:
            sample_image = copy.deepcopy(Image.open(os.path.join(config.get_value("TRAIN_IMAGES_PATH"), image_name+".jpg")).resize((config.get_value("IMAGE_WIDTH"), config.get_value("IMAGE_WIDTH"))))
            augmented_image = augmentation_transform(image=np.array(sample_image))
            augmented_images.append(Image.fromarray(augmented_image["image"]))

    normal_images_sampling = []
    for image_name in tqdm(sampled_ids):
        normal_images_sampling.append(copy.deepcopy(Image.open(os.path.join(config.get_value("TRAIN_IMAGES_PATH"), image_name+".jpg")).resize((config.get_value("IMAGE_WIDTH"), config.get_value("IMAGE_WIDTH")))))

    total_images = np.vstack((augmented_images, normal_images_sampling))
    targets = np.vstack((np.ones(len(augmented_images)), np.zeros(len(normal_images_sampling))))

    seed = np.random.randint(0, 255)
    return np.random.RandomState(seed).shuffle(total_images), np.random.RandomState(seed).shuffle(targets)


def load_val_images(val_ids, val_target, config):
    val_images = []
    for image_name in val_ids:
        sample_image = copy.deepcopy(
            Image.open(os.path.join(config.get_value("TRAIN_IMAGES_PATH"), image_name + ".jpg")))
        val_images.append(sample_image)

    return val_images, val_target


def create_folds(isic_id, metadata, labels, config):
    stratified_kf = StratifiedKFold(n_splits=config.get_value("K_FOLDS"), shuffle=True)

    fold_config = {}
    for i in range(config.get_value("K_FOLDS")):
        fold_config[f"{i + 1}-train"] = {}
        fold_config[f"{i + 1}-val"] = {}

    for fold_index, indexes in enumerate(stratified_kf.split(isic_id, labels)):
        train_index, val_index = indexes

        fold_config[f"{fold_index + 1}-train"]["isic_id"] = isic_id[train_index]
        fold_config[f"{fold_index + 1}-train"]["metadata"] = metadata[train_index]
        fold_config[f"{fold_index + 1}-train"]["target"] = labels[train_index]
        fold_config[f"{fold_index + 1}-val"]["isic_id"] = isic_id[val_index]
        fold_config[f"{fold_index + 1}-val"]["metadata"] = metadata[val_index]
        fold_config[f"{fold_index + 1}-val"]["target"] = labels[val_index]

    return fold_config