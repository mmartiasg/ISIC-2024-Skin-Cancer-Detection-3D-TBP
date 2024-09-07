import torch
from tqdm.auto import tqdm
import numpy as np
import torch.utils.data
from torcheval.metrics import Mean
import math
import os
import logging
import pandas as pd
from sklearn.metrics import roc_auc_score
import operator

"""
2024 ISIC Challenge primary prize scoring metric

Given a list of binary labels, an associated list of prediction 
scores ranging from [0,1], this function produces, as a single value, 
the partial area under the receiver operating characteristic (pAUC) 
above a given true positive rate (TPR).
https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

(c) 2024 Nicholas R Kurtansky, MSKCC
"""
from sklearn.metrics import roc_curve, auc


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float = 0.80) -> float:
    '''
    2024 ISIC Challenge metric: pAUC

    Given a solution file and submission file, this function returns the
    the partial area under the receiver operating characteristic (pAUC)
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        solution: ground truth pd.DataFrame of 1s and 0s
        submission: solution dataframe of predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # check submission is numeric
    if not pd.api.types.is_numeric_dtype(submission.values):
        raise Exception('Submission target column must be numeric')

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution.values) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(submission.values)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return (partial_auc)


class EarlyStopping:
    def __init__(self, tolerance=5, direction="min"):
        self.tolerance = tolerance
        self.counter = 0
        self.comparison_operator = operator.gt if direction == "max" else operator.lt
        self.best_validation_loss = float('inf') if direction == "min" else 0.0

    def __call__(self, validation_loss):
        if self.comparison_operator(validation_loss, self.best_validation_loss):
            self.best_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter > self.tolerance


class SaveBestModel:
    def __init__(self, path, logger, version, direction):
        self.path = path
        self.logger = logger
        self.best_validation_loss = float('inf') if direction == "min" else -float('inf')
        self.version = version
        self.direction = operator.gt if direction == 'max' else operator.lt
        os.makedirs(self.path, exist_ok=True)

    def save_model(self, model):
        #Save weights
        torch.save(model.state_dict(), os.path.join(self.path, f"{self.version}_best.pt"))

    def __call__(self, validation_loss, model):
        if self.direction(validation_loss, self.best_validation_loss):
            self.save_model(model)
            self.logger.info(
                f"Saved model's weight improvement from {self.best_validation_loss} to {validation_loss}")
            self.best_validation_loss = validation_loss


class MixUpV1:
    def __init__(self, alpha=0.2):
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, batch):
        """Returns mixed inputs, pairs of targets, and lambda"""
        x1, y = torch.utils.data.default_collate(batch)
        lam = self.beta_dist.sample()
        index = torch.randperm(x1.size()[0])

        return lam * x1 + (1 - lam) * x1[index, :], y, y[index], lam


class MixUpV2:
    def __init__(self, alpha=0.2):
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, batch):
        """Returns mixed inputs, pairs of targets, and lambda"""
        x1, x2, y = torch.utils.data.default_collate(batch)
        lam = self.beta_dist.sample()
        index = torch.randperm(x1.size()[0])

        return lam * x1 + (1 - lam) * x1[index, :], y, y[index], lam, lam * x2 + (1 - lam) * x2[index, :]


def roc_auc_metric(y_pred, y_true):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)

    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_pred])

    partial_auc_scaled = roc_auc_score(y_true=v_gt, y_score=v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return torch.tensor(partial_auc)


def train_single_task_v1(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs, config):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'results/{config.get_value("VERSION")}_{config.get_value("MODEL")}_training.log',
                        encoding='utf-8', level=logging.INFO)

    train_history_epoch_loss = []
    val_history_epoch_loss = []
    val_metric_epoch = []

    save_best_model = SaveBestModel("checkpoint_resnet50_mix_up",
                                    version=f'{config.get_value("VERSION")}_{config.get_value("MODEL")}', logger=logger,
                                    direction="max")
    early_stopping = EarlyStopping(tolerance=config.get_value("TOLERANCE_EARLY_STOPPING"), direction="max")

    scheduler = None
    if config.get_value("ENABLED_EXPONENTIAL_LR"):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    train_epoch_loss = Mean().to("cpu")
    val_epoch_loss = Mean().to("cpu")
    val_metric = Mean().to("cpu")

    train_loss_epoch_value = val_loss_epoch_value = val_metric_epoch_value = 0

    with tqdm(total=epochs) as epoch_p_bar:
        for epoch in range(epochs):
            train_batch_processed = 1

            model.train()
            for train_batch in train_dataloader:

                if config.get_value("USING_MIXUP"):
                    mix_x, y_a, y_b, lam = (train_batch[0].to(device=device, dtype=torch.float),
                                            train_batch[1].to(device=device, dtype=torch.float),
                                            train_batch[2].to(device=device, dtype=torch.float),
                                            train_batch[3].to(device=device, dtype=torch.float))
                else:
                    mix_x, y = (train_batch[0].to(device=device, dtype=torch.float),
                                train_batch[1].to(device=device, dtype=torch.float))

                train_batch_len = len(mix_x)

                optimizer.zero_grad(set_to_none=True)

                y_train_pred = model(mix_x)

                if config.get_value("USING_MIXUP"):
                    train_loss = (criterion(input=y_train_pred, target=y_a) * lam +
                                  criterion(input=y_train_pred, target=y_b) * (1 - lam))
                else:
                    train_loss = criterion(input=y_train_pred, target=y)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss.update(train_loss.detach().cpu() * train_batch_len)
                train_loss_epoch_value = train_epoch_loss.compute().item()
                metric_update = (
                    f"[Train loss: {round(train_loss_epoch_value, 4)}] - [Val loss: {round(val_loss_epoch_value, 4)}] - [Val Metric RoC>0.8: {round(val_metric_epoch_value, 4)}")

                epoch_p_bar.set_description(
                    f"{metric_update} | Train batch processed: {train_batch_processed} - {len(train_dataloader)} - {math.ceil(train_batch_processed / len(train_dataloader) * 100)}%")
                train_batch_processed += 1

            # Validation
            model.eval()
            y_val_preds_list = []
            y_val_list = []
            with torch.no_grad():
                val_batch_processed = 1
                for val_batch in val_dataloader:
                    x_val, y_val = (val_batch[0].to(device=device, dtype=torch.float),
                                    val_batch[1].to(device=device, dtype=torch.float))

                    val_batch_len = len(x_val)

                    y_val_pred = model(x_val)

                    val_loss = criterion(input=y_val_pred, target=y_val)

                    val_epoch_loss.update(val_loss.cpu() * val_batch_len)
                    val_loss_epoch_value = val_epoch_loss.compute().item()

                    y_val_preds_list.append(y_val_pred.cpu().numpy())
                    y_val_list.append(y_val.cpu().numpy())

                    metric_update = (
                        f"[Train loss: {round(train_loss_epoch_value, 4)}] - [Val loss: {round(val_loss_epoch_value, 4)}] - [Val Metric RoC>0.8: {round(val_metric_epoch_value, 4)}")

                    epoch_p_bar.set_description(
                        f"{metric_update} | Val batch processed: {val_batch_processed} - {len(val_dataloader)} - {math.ceil(val_batch_processed / len(val_dataloader) * 100)}%")
                    val_batch_processed += 1

                val_metric.update(roc_auc_metric(np.vstack(y_val_preds_list), np.vstack(y_val_list)))
                val_metric_epoch_value = val_metric.compute().item()

            #Clean metrics state at the end of the epoch
            train_epoch_loss.reset()
            val_epoch_loss.reset()
            val_metric.reset()
            #Scheduler step
            if scheduler:
                scheduler.step()

            train_history_epoch_loss.append(train_loss_epoch_value)
            val_history_epoch_loss.append(val_loss_epoch_value)
            val_metric_epoch.append(val_metric_epoch_value)

            save_best_model(validation_loss=val_metric_epoch_value, model=model)
            if early_stopping(validation_loss=val_metric_epoch_value):
                logger.info("Stopped early")
                break

            epoch_p_bar.update(1)

    return train_history_epoch_loss, val_history_epoch_loss, val_metric_epoch


def train_single_task_v2(model, train_dataloader, val_dataloader, optimizer, criterion_main, criterion_aux, contribution_criterion, device, epochs, config):
    logger = logging.getLogger(__name__)

    logging.basicConfig(filename=f'results/{config.get_value("VERSION")}_{config.get_value("MODEL")}_training.log',
                        encoding='utf-8', level=logging.INFO)

    train_history_epoch_loss = []
    val_history_epoch_loss = []
    val_metric_epoch = []

    save_best_model = SaveBestModel("checkpoint_resnet50_mix_up",
                                    version=f'{config.get_value("VERSION")}_{config.get_value("MODEL")}', logger=logger,
                                    direction="max")
    early_stopping = EarlyStopping(tolerance=config.get_value("TOLERANCE_EARLY_STOPPING"), direction="max")

    scheduler = None
    if config.get_value("ENABLED_EXPONENTIAL_LR"):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    train_epoch_loss = Mean().to("cpu")
    val_epoch_loss = Mean().to("cpu")
    val_metric = Mean().to("cpu")

    train_loss_epoch_value = val_loss_epoch_value = val_metric_epoch_value = 0

    with tqdm(total=epochs) as epoch_p_bar:
        for epoch in range(epochs):
            train_batch_processed = 1

            model.train()
            for train_batch in train_dataloader:

                if config.get_value("USING_MIXUP"):
                    mix_x, y_a, y_b, lam, metadata_x = (train_batch[0].to(device=device, dtype=torch.float),
                                                        train_batch[1].to(device=device, dtype=torch.float),
                                                        train_batch[2].to(device=device, dtype=torch.float),
                                                        train_batch[3].to(device=device, dtype=torch.float),
                                                        train_batch[4].to(device=device, dtype=torch.float))
                else:
                    mix_x, y = (train_batch[0].to(device=device, dtype=torch.float),
                                train_batch[1].to(device=device, dtype=torch.float))

                train_batch_len = len(mix_x)

                optimizer.zero_grad(set_to_none=True)

                y_train_pred = model([mix_x, metadata_x])

                if config.get_value("USING_MIXUP"):
                    train_loss = ((criterion_main(input=y_train_pred, target=y_a) * lam +
                                    criterion_main(input=y_train_pred, target=y_b) * (1 - lam)) * contribution_criterion
                                  +
                                    criterion_aux(input=y_train_pred, target=y_a) * lam +
                                    criterion_aux(input=y_train_pred, target=y_b) * (1 - lam) * (1 - contribution_criterion)
                                  )
                else:
                    train_loss = criterion_main(input=y_train_pred, target=y) * contribution_criterion + criterion_aux(input=y_train_pred, target=y) * (1 - contribution_criterion)

                train_loss.backward()

                optimizer.step()

                train_epoch_loss.update(train_loss.detach().cpu() * train_batch_len)
                train_loss_epoch_value = train_epoch_loss.compute().item()
                # metric_update = (
                #     f"[Train loss: {round(train_loss_epoch_value, 4)}] - [Val loss: {round(val_loss_epoch_value, 4)}] - [Val Metric RoC>0.8: {round(val_metric_epoch_value, 4)}")
                metric_update = (
                    f"[Train loss: {round(train_loss_epoch_value, 4)}] - [Val Metric RoC>0.8: {round(val_metric_epoch_value, 4)}]")

                epoch_p_bar.set_description(
                    f"{metric_update} | Train batch processed: {train_batch_processed} - {len(train_dataloader)} - {math.ceil(train_batch_processed / len(train_dataloader) * 100)}%")
                train_batch_processed += 1

            # Validation
            model.eval()
            y_val_preds_list = []
            y_val_list = []
            with torch.no_grad():
                val_batch_processed = 1
                for val_batch in val_dataloader:
                    x1_val, x2_val, y_val = (val_batch[0].to(device=device, dtype=torch.float),
                                             val_batch[1].to(device=device, dtype=torch.float),
                                             val_batch[2].to(device=device, dtype=torch.float))

                    val_batch_len = len(x1_val)

                    y_val_pred = model([x1_val, x2_val])

                    # val_loss = criterion_main(input=y_val_pred, target=y_val) * contribution_criterion + criterion_aux(input=y_val_pred, target=y_val) * (1 - contribution_criterion)
                    # val_epoch_loss.update(val_loss.cpu() * val_batch_len)

                    val_loss_epoch_value = val_epoch_loss.compute().item()

                    y_val_preds_list.append(y_val_pred.cpu().numpy())
                    y_val_list.append(y_val.cpu().numpy())

                    # metric_update = (
                    #     f"[Train loss: {round(train_loss_epoch_value, 4)}] - [Val loss: {round(val_loss_epoch_value, 4)}] - [Val Metric RoC>0.8: {round(val_metric_epoch_value, 4)}")

                    metric_update = (
                        f"[Train loss: {round(train_loss_epoch_value, 4)}] - [Val Metric RoC>0.8: {round(val_metric_epoch_value, 4)}]")

                    epoch_p_bar.set_description(
                        f"{metric_update} | Val batch processed: {val_batch_processed} - {len(val_dataloader)} - {math.ceil(val_batch_processed / len(val_dataloader) * 100)}%")
                    val_batch_processed += 1

                val_metric.update(roc_auc_metric(np.vstack(y_val_preds_list), np.vstack(y_val_list)))
                val_metric_epoch_value = val_metric.compute().item()

            #Clean metrics state at the end of the epoch
            # train_epoch_loss.reset()
            val_epoch_loss.reset()
            val_metric.reset()
            #Scheduler step
            if scheduler:
                scheduler.step()

            train_history_epoch_loss.append(train_loss_epoch_value)
            # val_history_epoch_loss.append(val_loss_epoch_value)
            val_metric_epoch.append(val_metric_epoch_value)

            save_best_model(validation_loss=val_metric_epoch_value, model=model)
            if early_stopping(validation_loss=val_metric_epoch_value):
                logger.info("Stopped early")
                break

            epoch_p_bar.update(1)

    return train_history_epoch_loss, val_history_epoch_loss, val_metric_epoch
