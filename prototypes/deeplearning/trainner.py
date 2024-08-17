import torch
from tqdm.auto import tqdm
import numpy as np
from tqdm.auto import tqdm
import numpy as np
import torch.utils.data
from torcheval.metrics import MulticlassAccuracy, Mean
import math
import os
import logging
from torcheval.metrics import BinaryAUROC


class EarlyStopping:
    def __init__(self, tolerance=5):
        self.tolerance = tolerance
        self.counter = 0
        self.best_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter > self.tolerance


class SaveBestModel:
    def __init__(self, path, model, logger):
        self.path = path
        self.model = model
        self.logger = logger
        self.best_validation_loss = float('inf')
        os.makedirs(self.path, exist_ok=True)

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.path, "best.pt"))

    def __call__(self, validation_loss):
        if validation_loss < self.best_validation_loss:
            self.save_model()
            self.logger.info(
                f"Saved model's weight improvement from {self.best_validation_loss} to {validation_loss}")
            self.best_validation_loss = validation_loss


def train_model_dual_task(model, train_dataloader, val_dataloader, criterion_1, criterion_2, optimizer, device, epochs, contribution_percentage=0.5, use_log=False):
    """Training model logic. For this model I'm combining 2 losses from the classifier and the auxiliar.

        model (torch.nn.Module): model to train.
        train_dataloader(torch.utils.data.DataLoader): train data.
        val_dataloader(torch.utils.data.DataLoader): validation data.
        device(str): device to run the model.
        epochs(int): number of epochs to train the model.
        learning_rate(float): learning rate for the optimizer to adjust the weights.

        returns: trained model and history dictionary with loss and metrics values means and std per epoch.
    """

    model.to(device=device)

    train_loss_history = []
    val_loss_history = []
    last_val_loss_mean = 0

    with tqdm(total=epochs, position=0, leave=True) as epoch_bar:
        for epoch in range(epochs):
            loss_value_train = []
            loss_value_val = []

            model.train(True)
            for batch in train_dataloader:
                x = batch[0].to(device=device)
                y_1 = batch[1][0].to(device=device, dtype=torch.long)
                y_2 = batch[1][1].to(device=device, dtype=torch.float)

                y_pred = model(x)
                y_pred_task_classification = y_pred["classification"]
                y_pred_task_landmarks = y_pred["landmarks"]

                #Input and target
                loss_task_1 = criterion_1(y_pred_task_classification, y_1)
                # Input and target
                loss_task_2 = criterion_2(y_pred_task_landmarks, y_2)

                optimizer.zero_grad()

                # Combining losses giving more importance to the classifier than the auxiliar.
                if use_log:
                    combined_loss = (contribution_percentage * loss_task_1) + (1-contribution_percentage) * torch.log(loss_task_2)
                else:
                    combined_loss = (contribution_percentage * loss_task_1) + (1 - contribution_percentage) * loss_task_2

                combined_loss.backward()

                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                loss_value_train.append(combined_loss.item() * len(batch))

                epoch_bar.set_description(
                    'loss: {:.4f} | val_loss: {:.4f}'.format(
                        np.mean(loss_value_train),
                        last_val_loss_mean)
                )

            # Evaluate on validation
            model.train(False)
            for val_batch in val_dataloader:
                val_x = val_batch[0].to(device=device)
                val_y_1 = val_batch[1][0].to(device=device, dtype=torch.long)
                val_y_2 = val_batch[1][1].to(device=device, dtype=torch.float)

                val_y_pred = model(val_x)

                y_pred_val_task_classification = val_y_pred["classification"]
                y_pred_val_task_landmarks = val_y_pred["landmarks"]

                # Input and target
                loss_val_task_1 = criterion_1(y_pred_val_task_classification, val_y_1)
                # Input and target
                loss_val_task_2 = criterion_2(y_pred_val_task_landmarks, val_y_2)

                if use_log:
                    combined_loss = (contribution_percentage * loss_val_task_1) + ((1-contribution_percentage) * torch.log(loss_val_task_2))
                else:
                    combined_loss = (contribution_percentage * loss_val_task_1) + (
                                (1 - contribution_percentage) * loss_val_task_2)

                loss_value_val.append(combined_loss.item() * len(val_batch))

            last_val_loss_mean = np.mean(loss_value_val)
            val_loss_history.append(last_val_loss_mean)

            # Average from all batches
            train_loss_history.append(np.mean(loss_value_train))

            epoch_bar.update(1)

    return {"train_loss_history": train_loss_history, "val_loss_history": val_loss_history}


@torch.compile
def mix_up_data(x, y, alpha, beta_dist=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        # Will this make it faster?
        lam = beta_dist.sample()
        # lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # lam = beta_dist.sample()
    # batch_size = x.size()[0]
    index = torch.randperm(x.size()[0])

    # mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = y, y[index]

    # return mixed_x, y_a, y_b, lam
    return lam * x + (1 - lam) * x[index, :], y, y[index], lam


def train_single_task(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs, alpha=0.2):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='results/training.log', encoding='utf-8', level=logging.INFO)

    train_history_epoch_loss = []
    val_history_epoch_loss = []
    metric_update = "No Metrics so far.."

    save_best_model = SaveBestModel("checkpoint_resnet50_mix_up", model=model, logger=logger)
    early_stopping = EarlyStopping(tolerance=5)

    beta_dist = torch.distributions.beta.Beta(alpha, alpha)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_epoch_loss = Mean().to("cpu")
    val_epoch_loss = Mean().to("cpu")
    val_metric = BinaryAUROC().to("cpu")
    train_metric = BinaryAUROC().to("cpu")

    with tqdm(total=epochs) as epoch_p_bar:
        for epoch in range(epochs):
            train_batch_processed = 1
            model.train()
            for train_batch in train_dataloader:
                mix_x, y_a, y_b, lam = mix_up_data(train_batch[0],
                                                   train_batch[1],
                                                   alpha,
                                                   beta_dist)

                mix_x, y_a, y_b = (mix_x.to(device=device, dtype=torch.float),
                                   y_a.to(device=device, dtype=torch.float),
                                   y_b.to(device=device, dtype=torch.float))

                train_batch_len = len(mix_x)

                y_train_pred = model(mix_x).to(device=device)

                train_loss = (criterion(input=y_train_pred, target=y_a) * lam +
                              criterion(input=y_train_pred, target=y_b) * (1-lam))

                # optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_epoch_loss.update(train_loss.detach().cpu() * train_batch_len)
                    train_metric.update(y_train_pred.squeeze().detach().cpu(), train_batch[1].squeeze())
                model.train()

                epoch_p_bar.set_description(f"{metric_update} | Train batch processed: {train_batch_processed} \
                - {len(train_dataloader)} - {math.ceil(train_batch_processed / len(train_dataloader) * 100)}%")
                train_batch_processed += 1


            model.eval()
            with torch.no_grad():
                val_batch_processed = 1
                for val_batch in val_dataloader:
                    x_val, y_val = (val_batch[0].to(device=device, dtype=torch.float),
                                    val_batch[1].to(device=device, dtype=torch.float))

                    val_batch_len = len(x_val)

                    y_val_pred = model(x_val)

                    val_loss = criterion(input=y_val_pred, target=y_val)

                    val_metric.update(y_val_pred.squeeze().detach().cpu(), y_val.squeeze())
                    val_epoch_loss.update(val_loss.detach().cpu() * val_batch_len)

                    epoch_p_bar.set_description(
                        f"{metric_update} | Val batch processed: {val_batch_processed} from {len(val_dataloader)} \
                        - {math.ceil(val_batch_processed / len(val_dataloader) * 100)}%")
                    val_batch_processed += 1
            model.train()

            train_history_epoch_loss.append(train_epoch_loss.compute().item())
            val_history_epoch_loss.append(val_epoch_loss.compute().item())

            metric_update = (f"[loss: {round(train_history_epoch_loss[-1], 4)} \
            - val_loss: {round(val_history_epoch_loss[-1], 4)}] - Train_rocAUC: {train_metric.compute().item()} Val_rocAUC: {val_metric.compute().item()}")

            # epoch_p_bar.set_description(f"[loss: {round(train_history_epoch_loss[-1], 4)} - val_loss: {round(val_history_epoch_loss[-1], 4)}]")

            #Clean metrics state at the end of the epoch
            train_epoch_loss.reset()
            val_epoch_loss.reset()
            train_metric.reset()
            val_metric.reset()

            #Scheduler step
            scheduler.step()

            save_best_model(validation_loss = val_history_epoch_loss[-1])
            if early_stopping(validation_loss = val_history_epoch_loss[-1]):
                logger.info("Stopped early")
                break

            epoch_p_bar.update(1)

    return train_history_epoch_loss, val_history_epoch_loss
