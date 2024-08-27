import gc

from prototypes.deeplearning.dataloader.IsicDataLoader import IsicDataLoader, over_under_sample, load_val_images, create_folds, AugmentationWrapper
from prototypes.deeplearning.trainner import train_single_task
import torch
import json
from prototypes.utility.data import ProjectConfiguration
import torchvision
import os
import matplotlib.pyplot as plt
from prototypes.deeplearning.trainner import MixUp
from prototypes.deeplearning.models import (Resnet50Prototype1,
                                            Resnet50Prototype2,
                                            Resnet50Prototype3,
                                            Resnet50Prototype1Dropout,
                                            Resnet50Prototype2Dropout,
                                            Resnet50Prototype3Dropout,
                                            VitPrototype1Dropout,
                                            VitPrototype2Dropout,
                                            VitPrototype3MHA)
from tqdm.auto import tqdm
import pandas as pd
from prototypes.deeplearning.trainner import score
import logging
import albumentations as A
import numpy as np


model_selection = {"prototype1": Resnet50Prototype1,
                   "prototype2": Resnet50Prototype2,
                   "prototype3": Resnet50Prototype3,
                   "prototype1Dropout": Resnet50Prototype1Dropout,
                   "prototype2Dropout": Resnet50Prototype2Dropout,
                   "prototype3Dropout": Resnet50Prototype3Dropout,
                   "vit16Dropout": VitPrototype1Dropout,
                   "vit16MixDropout": VitPrototype2Dropout,
                   "vit16MHA": VitPrototype3MHA}

def score_model(config, dataloader):
    model_loaded = model_selection[config.get_value("MODEL")](n_classes=config.get_value("NUM_CLASSES"))
    model_loaded.load_state_dict(
        torch.load(os.path.join("checkpoint_resnet50_mix_up", f"{config.get_value('VERSION')}_{config.get_value('MODEL')}_best.pt"), weights_only=True))

    model_loaded.eval()
    model_loaded = model_loaded.cuda()

    y_pred = []
    y_true = []

    for batch in tqdm(dataloader):
        x = batch[0].cuda()
        metadata_x = batch[1].cuda()
        y = batch[2].cuda()

        with torch.no_grad():
            y_pred.extend(model_loaded([x.float(), metadata_x.float()]).cpu().numpy())
            y_true.extend(y.cpu().numpy())

    solution_df = pd.DataFrame(zip(y_true, [e[0] for e in y_true]), columns=['isic_id', 'target'])
    submission_df = pd.DataFrame(zip(y_pred, [e[0] for e in y_pred]), columns=['isic_id', 'target'])

    return score(solution=solution_df, submission=submission_df, row_id_column_name="isic_id", min_tpr=0.80)


def main():
    config = ProjectConfiguration("config.json")

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'results/{config.get_value("VERSION")}_{config.get_value("MODEL")}_scores.log', encoding='utf-8', level=logging.INFO)

    model = model_selection[config.get_value("MODEL")](n_classes=config.get_value("NUM_CLASSES"))
    model = model.to(device=config.get_value("TRAIN_DEVICE"))

    os.makedirs(os.path.join("results", config.get_value("VERSION")), exist_ok=True)

    #Augmentation per sample
    augmentations = A.Compose([
        A.CLAHE(p=0.4),
        A.RandomRotate90(p=0.7),
        A.Transpose(p=0.6),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.5),
        A.HueSaturationValue(p=0.5),
    ])

    augmentation_transform_pipeline = torchvision.transforms.Compose([AugmentationWrapper(augmentations), model.weights.transforms])

    # Augmentation cross sample
    mix_up = MixUp(alpha=config.get_value("ALPHA"))

    metadata_df = pd.read_csv(config.get_value("TRAIN_METADATA"), engine="python")
    columns = config.get_value("METADATA_COLUMNS").split("\t")

    isic_id, metadata_array, labels = metadata_df["isic_id"].values, metadata_df[columns].values, metadata_df[
        "target"].values

    folds_config_dict = create_folds(isic_id=isic_id, metadata=metadata_array, labels=labels, config=config)

    total_val_history = []
    total_score = []
    print(folds_config_dict.keys())

    for fold_index in folds_config_dict.keys():
        print(f"Fold {fold_index}")

        train_x, train_y = over_under_sample(anomaly_images=folds_config_dict[fold_index]["train"]["isic_id"][np.where(folds_config_dict[fold_index]["train"]["target"]==1)],
                          normal_images=folds_config_dict[fold_index]["train"]["isic_id"][np.where(folds_config_dict[fold_index]["train"]["target"]==0)],
                          config=config,
                          augmentation_transform=augmentations,
                          total_samples=config.get_value("TOTAL_TRAIN_SAMPLES"),
                          imbalance_percentage=config.get_value("CLASS_BALANCE_PERCENTAGE"))

        val_x, val_y = load_val_images(val_ids=folds_config_dict[fold_index]["val"]["isic_id"],
                                       val_target=folds_config_dict[fold_index]["val"]["target"], config=config)

        train = IsicDataLoader(train_x, train_y, transform=augmentation_transform_pipeline)
        val = IsicDataLoader(val_x, val_y, transform=model.weights.transforms)

        train_sampler = val_sampler = None
        shuffle = True
        if config.get_value("USE_SAMPLER"):
            train_sampler = torch.utils.data.RandomSampler(train, num_samples=config.get_value("TRAIN_SAMPLE_SIZE"))
            val_sampler = torch.utils.data.RandomSampler(val, num_samples=config.get_value("VAL_SAMPLE_SIZE"))
            shuffle = False

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=config.get_value("BATCH_SIZE"),
                                                       shuffle=shuffle,
                                                       num_workers=config.get_value("NUM_WORKERS"),
                                                       pin_memory=True,
                                                       sampler=train_sampler,
                                                       collate_fn=mix_up,
                                                       prefetch_factor=config.get_value("PREFETCH_FACTOR"),
                                                       persistent_workers=False)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=config.get_value("BATCH_SIZE"),
                                                     shuffle=False,
                                                     num_workers=config.get_value("NUM_WORKERS"),
                                                     pin_memory=True,
                                                     sampler=val_sampler,
                                                     prefetch_factor=config.get_value("PREFETCH_FACTOR"),
                                                     persistent_workers=False)

        print(f"Train set size: {len(train_dataloader.dataset)}\
         | Validation set size: {len(val_dataloader.dataset)}")

        train_history, val_history = train_single_task(model=model,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          optimizer=torch.optim.Adam(params=model.parameters(), lr=config.get_value("LEARNING_RATE")),
                          criterion=torch.nn.BCELoss(),
                          device=config.get_value("TRAIN_DEVICE"),
                          epochs=config.get_value("NUM_EPOCHS"),
                          config=config)

        plt.plot(range(len(train_history)), train_history, label=f"Training Loss fold: {fold_index}")
        plt.plot(range(len(val_history)), val_history, label=f"Validation Loss fold: {fold_index}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join("results", config.get_value("VERSION"), f"{config.get_value('MODEL')}_Training_val_loss_curves_{fold_index}.png"))

        # free up ram and vram
        del model
        torch.cuda.empty_cache()
        gc.collect()


        score = score_model(config, dataloader=val_dataloader)
        logger.info(f"Model version: {config.get_value('VERSION')}_{config.get_value('MODEL')} score a : {score} in the validation dataset")

        total_score.append(score)
        total_val_history.extend(val_history)

    print(f"Total score mean and std: {np.mean(total_score)} | {np.std(total_score)}")
    print(f"Total val score mean and std: {np.mean(total_val_history)} | {np.std(total_val_history)}")


if __name__ == "__main__":
    main()
