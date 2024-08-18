import torch
import h5py
from PIL import Image
import pandas as pd
import io
import numpy as np
import glob
import os
import re


class LoadDataVectors(torch.utils.data.Dataset):
    def __init__(self, hd5_file_path, metadata_csv_path=None, transform=None, target_transform=None):
        super(LoadDataVectors, self).__init__()

        self.hd5_file = h5py.File(hd5_file_path, "r")
        self.keys = list(self.hd5_file.keys())
        self.transform = transform
        self.metadata_dataframe = None
        if metadata_csv_path is not None:
            self.metadata_dataframe = pd.read_csv(metadata_csv_path, engine="python")
            self.target_dict = dict(zip(self.metadata_dataframe["isic_id"].values, self.metadata_dataframe["target"].values))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        x = Image.open(io.BytesIO(self.hd5_file[self.keys[idx]][()]))

        if self.transform:
            x = self.transform(x)

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