import torch
import h5py
from PIL import Image
import pandas as pd
import io
import numpy as np


class LoadDataVectors(torch.utils.data.Dataset):
    def __init__(self, hd5_file_path, metadata_csv_path, target_columns, transform=None, target_transform=None):
        self.hd5_file = h5py.File(hd5_file_path, "r")
        self.keys = list(self.hd5_file.keys())
        self.transform = transform
        self.metadata_dataframe = pd.read_csv(metadata_csv_path, engine="python")
        self.target_columns = target_columns

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        dataset = self.hd5_file[self.keys[idx]]
        image_arr_bytes = dataset[()]
        image = Image.open(io.BytesIO(image_arr_bytes))
        x = image

        if self.transform:
            x = self.transform.transform(np.array(image))

        if self.metadata_dataframe is not None:
            return x,\
                self.metadata_dataframe[self.metadata_dataframe["isic_id"] == self.keys[idx]]\
                    [self.target_columns].to_numpy()

        return x
