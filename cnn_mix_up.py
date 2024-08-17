from prototypes.deeplearning.dataloader.IsicDataLoader import LoadDataVectors
from prototypes.deeplearning.trainner import train_single_task
import torch
import json
from prototypes.utility.data import ProjectConfiguration
import torchvision
import os
import matplotlib.pyplot as plt


def main():
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    config = ProjectConfiguration("config.json")

    os.makedirs(os.path.join("results", config.get_value("VERSION")), exist_ok=True)

    dataloader = LoadDataVectors(hd5_file_path=os.path.join(config.get_value("DATASET_PATH"), "train-image.hdf5"),
                                 metadata_csv_path=config.get_value("TRAIN_METADATA"),
                                 target_columns=["target"],
                                 transform=weights.transforms())

    train, val = torch.utils.data.random_split(dataloader,
                                               [config.get_value("TRAIN_SPLIT"), 1 - config.get_value("TRAIN_SPLIT")])

    train_sampler = val_sampler = None
    shuffle = True
    if config.get_value("USE_SAMPLER"):
        train_sampler = torch.utils.data.RandomSampler(train, num_samples=config.get_value("TRAIN_SAMPLE_SIZE"))
        val_sampler = torch.utils.data.RandomSampler(val, num_samples=config.get_value("VAL_SAMPLE_SIZE"))
        shuffle = False

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config.get_value("BATCH_SIZE"), shuffle=shuffle,
                                                   num_workers=config.get_value("NUM_WORKERS"), pin_memory=True, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config.get_value("BATCH_SIZE"), shuffle=False,
                                                 num_workers=config.get_value("NUM_WORKERS"), pin_memory=True, sampler=val_sampler)


    print(f"Train set size: {len(train)} | Validation set size: {len(val)}")

    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, config.get_value("NUM_CLASSES")), torch.nn.Sigmoid())

    print("Freezing parameters...")
    for param in model.layer1.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = False

    model = model.to(device=config.get_value("TRAIN_DEVICE"))

    train_history, val_history = train_single_task(model=model,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-4),
                      criterion=torch.nn.BCELoss(),
                      device=config.get_value("TRAIN_DEVICE"),
                      epochs=config.get_value("NUM_EPOCHS"),
                      alpha=config.get_value("ALPHA"))

    plt.plot(train_history, len(train_history), label="Training Loss")
    plt.savefig(os.path.join("results", config.get_value("VERSION"), "Training_loss_curve.png"))

    plt.plot(val_history, len(val_history), label="Validation Loss")
    plt.savefig(os.path.join("results", config.get_value("VERSION"), "Validation_loss_curve.png"))

if __name__ == "__main__":
    main()
