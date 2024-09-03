import gc

from prototypes.deeplearning.dataloader.IsicDataLoader import IsicDataLoaderFolders, AugmentationWrapper
from prototypes.deeplearning.trainner import train_single_task_v1, train_single_task_v2
import torch
from prototypes.utility.data import ProjectConfiguration
import torchvision
import os
import matplotlib.pyplot as plt
from prototypes.deeplearning.trainner import MixUpV1, MixUpV2
from prototypes.deeplearning.models import (Resnet50Prototype1,
                                            Resnet50Prototype2,
                                            Resnet50Prototype3,
                                            Resnet50Prototype1Dropout,
                                            Resnet50Prototype2Dropout,
                                            Resnet50Prototype3Dropout,
                                            VitPrototype1Dropout,
                                            VitPrototype2Dropout,
                                            Vit_b_16_MHA,
                                            Vit16,
                                            MaxVit,
                                            SwingB,
                                            SwingV2B,
                                            ResNex10164x4d,
                                            WideResNet101)

from tqdm.auto import tqdm
import pandas as pd
from prototypes.deeplearning.trainner import score
import logging
import albumentations as A
import numpy as np
from prototypes.deeplearning.dataloader.IsicDataLoader import metadata_transform


model_selection = {"prototype1": Resnet50Prototype1,
                   "prototype2": Resnet50Prototype2,
                   "prototype3": Resnet50Prototype3,
                   "prototype1Dropout": Resnet50Prototype1Dropout,
                   "prototype2Dropout": Resnet50Prototype2Dropout,
                   "prototype3Dropout": Resnet50Prototype3Dropout,
                   "Vit16": Vit16,
                   "Vit16Dropout": VitPrototype1Dropout,
                   "vit16MixDropout": VitPrototype2Dropout,
                   "Vitb16MHA": Vit_b_16_MHA,
                   "MaxVit": MaxVit,
                   "SwingB": SwingB,
                   "SwingV2B": SwingV2B,
                   "ResNex10164x4d": ResNex10164x4d,
                   "WideResNet101": WideResNet101}


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
        y = batch[1].cuda()

        with torch.no_grad():
            y_pred.extend(model_loaded(x.float()).cpu().numpy())
            y_true.extend(y.cpu().numpy())

    solution_df = pd.DataFrame(zip(y_true, [e[0] for e in y_true]), columns=['isic_id', 'target'])
    submission_df = pd.DataFrame(zip(y_pred, [e[0] for e in y_pred]), columns=['isic_id', 'target'])

    return score(solution=solution_df, submission=submission_df, row_id_column_name="isic_id", min_tpr=0.80)


def main():
    config = ProjectConfiguration("config.json")

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'results/{config.get_value("VERSION")}_{config.get_value("MODEL")}_scores.log', encoding='utf-8', level=logging.INFO)
    os.makedirs(os.path.join("results", config.get_value("VERSION")), exist_ok=True)

    # Augmentation cross sample
    mix_up = MixUpV2(alpha=config.get_value("ALPHA"))

    folds = config.get_value("K_FOLDS")

    # Save values for all the folds
    total_score = total_val_history = []

    for fold_index in range(folds):
        print(f"Fold {fold_index + 1}")

        model = model_selection[config.get_value("MODEL")](n_classes=config.get_value("NUM_CLASSES"))
        model = model.to(device=config.get_value("TRAIN_DEVICE"))

        # Augmentation per sample
        augmentations = A.Compose([
            A.CLAHE(p=0.15),
            A.RandomRain(p=0.1, slant_lower=-10, slant_upper=10,
                         drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                         blur_value=5, brightness_coefficient=0.9, rain_type=None),
            A.MedianBlur(p=0.15),
            # A.ToGray(p=0.05),
            A.ImageCompression(quality_lower=55, p=0.15),
            # A.Equalize(p=0.15),
            A.ZoomBlur(p=0.15),
            A.GaussNoise(p=0.15),
            # A.RandomSnow(p=0.05),
            A.Sharpen(p=0.15),
            # A.ChromaticAberration(p=0.15),
            A.Rotate(limit=(-90, 90), p=0.25, crop_border=True),
            A.VerticalFlip(p=0.15),
            A.HorizontalFlip(p=0.15),
            A.Blur(blur_limit=3, p=0.1),
            A.OpticalDistortion(p=0.15),
            A.GridDistortion(p=0.15),
            # A.HueSaturationValue(p=0.15),
        ])

        augmentation_transform_pipeline = torchvision.transforms.Compose(
            [AugmentationWrapper(augmentations), model.weights.transforms()])

        train_val_metadata = metadata_transform(pd.read_csv(config.get_value("TRAIN_METADATA"), engine="python"))

        root_folder_train = os.path.join(config.get_value("DATASET_PATH"), "splits", f"fold_{fold_index + 1}", "train")
        train = IsicDataLoaderFolders(root=root_folder_train, transform=augmentation_transform_pipeline, metadata=train_val_metadata)

        root_folder_val = os.path.join(config.get_value("DATASET_PATH"), "splits", f"fold_{fold_index + 1}", "val")
        val = IsicDataLoaderFolders(root=root_folder_val, transform=model.weights.transforms(), metadata=train_val_metadata)

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
                                                       collate_fn=mix_up if config.get_value("USING_MIXUP") else None,
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

        train_history, val_history, metric_history = train_single_task_v2(model=model,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          optimizer=torch.optim.Adam(params=model.parameters(), lr=config.get_value("LEARNING_RATE")),
                          criterion=torch.nn.BCELoss(),
                          device=config.get_value("TRAIN_DEVICE"),
                          epochs=config.get_value("NUM_EPOCHS"),
                          config=config)

        plt.plot(range(len(train_history)), train_history, label=f"Training Loss fold: {fold_index + 1}")
        plt.plot(range(len(val_history)), val_history, label=f"Validation Loss fold: {fold_index + 1}")
        plt.plot(range(len(metric_history)), metric_history, label=f"Metric Val fold: {fold_index + 1}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join("results", config.get_value("VERSION"), f"{config.get_value('MODEL')}_Training_val_loss_metric_val_curves_{fold_index + 1}.png"))

        # free up ram and vram
        del model
        torch.cuda.empty_cache()
        gc.collect()

        #Do not need it anymore
        # score = score_model(config, dataloader=val_dataloader)
        # logger.info(f"Model version: {config.get_value('VERSION')}_{config.get_value('MODEL')} score a : {score} in the validation dataset in FOLD: {fold_index + 1}")

        total_score.append(score)
        total_val_history.extend(val_history)

    logger.info(f"Total score mean and std: {np.mean(total_score)} | {np.std(total_score)}")
    logger.info(f"Total val score mean and std: {np.mean(total_val_history)} | {np.std(total_val_history)}")


if __name__ == "__main__":
    main()
