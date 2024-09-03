import cv2
import torch
import h5py
import torchvision
from PIL import Image
import pandas as pd
import io
import numpy as np
import os
import re
from prototypes.classical.segmentation.transformers import OtsuThresholdingSegmentation, BlackBarsRemover
from sklearn.model_selection import StratifiedKFold
import math
import copy
from tqdm.auto import tqdm
import glob


def mixup_data(x, ids, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x.astype(np.uint8), ids[index]


def create_folds(isic_id, metadata, labels, config):
    stratified_kf = StratifiedKFold(n_splits=config.get_value("K_FOLDS"), shuffle=True)

    fold_config = {}
    for i in range(config.get_value("K_FOLDS")):
        fold_config[f"{i + 1}"] = {}
        fold_config[f"{i + 1}"]["train"] = {}
        fold_config[f"{i + 1}"]["val"] = {}

    for fold_index, indexes in enumerate(stratified_kf.split(isic_id, labels)):
        train_index, val_index = indexes

        fold_config[f"{fold_index + 1}"]["train"]["isic_id"] = isic_id[train_index]
        fold_config[f"{fold_index + 1}"]["train"]["metadata"] = metadata[train_index]
        fold_config[f"{fold_index + 1}"]["train"]["target"] = labels[train_index]
        fold_config[f"{fold_index + 1}"]["val"]["isic_id"] = isic_id[val_index]
        fold_config[f"{fold_index + 1}"]["val"]["metadata"] = metadata[val_index]
        fold_config[f"{fold_index + 1}"]["val"]["target"] = labels[val_index]

    return fold_config


class AugmentationWrapper():
    def __init__(self, augmentation_transform):
        self.augmentation_transform = augmentation_transform

    def __call__(self, sample):
        return Image.fromarray(self.augmentation_transform(image=np.array(sample))["image"])


class IsicDataLoaderMemory(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        super(IsicDataLoaderMemory, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]

        if self.transform:
            x = self.transform(x)

        return x, torch.tensor([self.y[idx]])


def metadata_transform(df, mean=None, std=None):
    new_df = df.copy()

    sex = new_df.sex.apply(lambda x: 0 if x == "male" else 1).values.astype(np.float32)
    age_ratio_size_lesion = (sex * new_df["clin_size_long_diam_mm"] * new_df["tbp_lv_symm_2axis"]).values.astype(np.float32)
    hue_contrast = np.abs((new_df['tbp_lv_H'] - new_df['tbp_lv_Hext']).values).astype(np.float32)
    luminance_contrast = np.abs((new_df['tbp_lv_L'] - new_df['tbp_lv_Lext']).values).astype(np.float32)
    lesion_color_difference = np.sqrt((new_df['tbp_lv_deltaA']**2 + new_df["tbp_lv_deltaB"]**2 + new_df['tbp_lv_deltaL']**2).values).astype(np.float32)
    border_complexity = (new_df['tbp_lv_norm_border'] + new_df['tbp_lv_symm_2axis']).values.astype(np.float32)
    color_uniformity = (new_df['tbp_lv_color_std_mean'] / (new_df['tbp_lv_radial_color_std_max'] + 1e-5)).values.astype(np.float32)

    ids = new_df['isic_id']

    features = np.vstack([age_ratio_size_lesion, hue_contrast, luminance_contrast, lesion_color_difference,
                          border_complexity, color_uniformity]).astype(np.float32)

    if mean and std:
        mean = np.mean(features, axis=1).reshape(-1, 1)
        std = np.std(features, axis=1).reshape(-1, 1)

    features -= mean
    features /= std

    return dict(zip(ids, features)), mean, std


class IsicDataLoaderFolders(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, metadata=None):
        self.transform = transform
        self.target_transform = target_transform
        self.classes = os.listdir(root)
        self.paths = []
        self.metadata_dict = metadata
        for n_class in self.classes:
            self.paths.extend(glob.glob(os.path.join(root, f"{n_class}", "*.jpg")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = Image.open(self.paths[idx])
        y = int(self.paths[idx].split(os.sep)[-2])

        if self.transform:
            x = self.transform(x)
        else:
            x = torchvision.transforms.ToTensor()(x)

        if self.target_transform:
            y = self.target_transform(y)
        else:
            y = torch.tensor([y])

        if self.metadata_dict:
            return x, y, torch.tensor(self.metadata_dict[self.paths[idx].split(os.sep)[-1].split("_")[0]])

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


def over_under_sample(anomaly_images_ids, normal_images_ids, augmentation_transform, root_path, config):
    if config.get_value("TOTAL_TRAIN_SAMPLES") is not None and config.get_value("TOTAL_TRAIN_SAMPLES") < anomaly_images_ids.shape[0] + normal_images_ids.shape[0]:
        total_samples = config.get_value("TOTAL_TRAIN_SAMPLES")
    else:
        total_samples = anomaly_images_ids.shape[0] + normal_images_ids.shape[0]

    imbalance_percentage = config.get_value("CLASS_BALANCE_PERCENTAGE")

    assert 0 < imbalance_percentage <= 0.5
    normal_image_percentage = 1 - imbalance_percentage

    print(f"Target: Normal samples count: {math.ceil(total_samples * normal_image_percentage)} | Target: Anomally samples count: {math.ceil(total_samples * imbalance_percentage)} ")

    iterations = math.ceil((total_samples * imbalance_percentage) / anomaly_images_ids.shape[0])

    print(f"Iterations needed to reach target count for anomaly samples: {iterations} | Anomaly original count: {anomaly_images_ids.shape[0]}")
    for iteration in tqdm(range(iterations)):

        image_buffer = np.zeros((len(anomaly_images_ids), config.get_value("IMAGE_WIDTH"),
                                 config.get_value("IMAGE_HEIGHT"), 3))

        for suffix_index, image_id in enumerate(anomaly_images_ids):
            sample_image = copy.deepcopy(Image.open(os.path.join(config.get_value("TRAIN_IMAGES_PATH"), image_id+".jpg")
                                                    ).resize((config.get_value("IMAGE_WIDTH"),
                                                              config.get_value("IMAGE_HEIGHT")))
                                         )

            augmented_image = augmentation_transform(image=np.array(sample_image))['image']
            image_buffer[suffix_index] = copy.deepcopy(augmented_image)

        mixed_images, mixed_ids = mixup_data(image_buffer, ids=anomaly_images_ids, alpha=0.8)

        for suffix_index, mixed in enumerate(zip(mixed_images, mixed_ids)):
            mixed_image, mixed_id = mixed
            Image.fromarray(mixed_image).save(os.path.join(root_path, "train", "1", f"{mixed_id}_{iteration}_{suffix_index}.jpg"))

    sampled_ids = np.random.choice(normal_images_ids, size=math.ceil(total_samples * normal_image_percentage), replace=False)
    print(f"Sampled normal images: {len(sampled_ids)} | unique numbers: {len(np.unique(sampled_ids))}")
    for image_id in tqdm(sampled_ids):
        # TODO: should I apply the same augmentation to normal images here or Is better to do it in the traning loop?
        Image.open(os.path.join(config.get_value("TRAIN_IMAGES_PATH"), image_id+".jpg"))\
            .resize((config.get_value("IMAGE_WIDTH"), config.get_value("IMAGE_WIDTH")))\
            .save(os.path.join(root_path, "train", "0", f"{image_id}.jpg"))


def load_val_images(val_ids, val_target, config):
    val_images = []
    for image_name in val_ids:
        sample_image = copy.deepcopy(
            Image.open(os.path.join(config.get_value("TRAIN_IMAGES_PATH"), image_name + ".jpg")))
        val_images.append(sample_image)

    return val_images, val_target
