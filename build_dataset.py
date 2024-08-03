import shutil
import multiprocessing as mpt

from prototypes.classical.dataloader.Loader import IsiCancerData
from prototypes.classical.descriptors.vetorizer import LBPVectorizer
import json
from prototypes.classical.descriptors.texture import GaborTransformer, LBPTransformer
import numpy as np
from tqdm.auto import tqdm
import os
import logging
from joblib import Parallel, delayed


logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)

with open("config.json", "r") as f:
    config = json.load(f)

dataset = IsiCancerData(config)

gabor_banks = []
for theta in [np.pi, np.pi/2, np.pi/4]:
    gabor_banks.append(GaborTransformer(frequency=1/100, theta=theta, sigma_x=5, sigma_y=5))

lbp_transformer = LBPTransformer(p=8, r=1, method="ror")
lbp_vectorizer = LBPVectorizer()

FEATURE_VECTOR_PATH = os.path.join(config["VECTORS_PATH"], "gabor_attention_maps", f"{config['IMAGE_WIDTH']}_{config['IMAGE_HEIGHT']}")

def process_batch(batch, gabor_banks, batch_index, feature_vector_path):

    features = np.zeros((dataset.batch_size, 255 * len(gabor_banks) * 3))
    targets = np.zeros((dataset.batch_size, 1))
    x_batch, y_batch = batch

    feature_vector_bank = np.zeros((len(gabor_banks), 255 * 3))
    with tqdm(total=len(gabor_banks) * x_batch.shape[0]) as pbar:
        for index, (x, y) in enumerate(zip(x_batch, y_batch)):
            for bank_index, gabor_transformer in enumerate(gabor_banks):
                x_imag = gabor_transformer.transform(x)[1]
                attention_map = x.copy()

                attention_map[:, :, 0] = attention_map[:, :, 0] * (x_imag > 0)
                attention_map[:, :, 1] = attention_map[:, :, 1] * (x_imag > 0)
                attention_map[:, :, 2] = attention_map[:, :, 2] * (x_imag > 0)

                lbp_map_channel_1 = lbp_transformer.transform(attention_map[:, :, 0])
                lbp_map_channel_2 = lbp_transformer.transform(attention_map[:, :, 1])
                lbp_map_channel_3 = lbp_transformer.transform(attention_map[:, :, 2])

                feature_vector_bank[bank_index] = np.hstack((lbp_vectorizer.transform(lbp_map_channel_1),
                                                             lbp_vectorizer.transform(lbp_map_channel_2),
                                                             lbp_vectorizer.transform(lbp_map_channel_3)))

            features[index] = np.hstack(feature_vector_bank)
            targets[index] = y
            pbar.update(1)

    np.save(os.path.join(feature_vector_path, "feature_vector_" + str(batch_index) + ".npy"), features)
    np.save(os.path.join(feature_vector_path, "label_" + str(batch_index) + ".npy"), targets)

    return batch_index

if os.path.exists(FEATURE_VECTOR_PATH):
    shutil.rmtree(FEATURE_VECTOR_PATH)

os.makedirs(FEATURE_VECTOR_PATH)

# it is not CPU intensive mostly I/O bounded thus x2 seems to be fine too.
cpu_count = mpt.cpu_count() - 1

with Parallel(n_jobs=cpu_count, return_as="generator_unordered", prefer="threads") as parallel_execution:
    res = parallel_execution((delayed(process_batch)(batch=batch,
                                                     gabor_banks=gabor_banks,
                                                     batch_index=batch_index,
                                                     feature_vector_path=FEATURE_VECTOR_PATH)
                              for batch_index, batch in enumerate(dataset.get_next_batch())
    ))

    for r in res:
        logger.info(f"Batche {r} saved!")
