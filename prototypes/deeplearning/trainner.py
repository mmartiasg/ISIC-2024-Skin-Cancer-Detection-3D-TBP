import torch
from tqdm.auto import tqdm
import numpy as np
import torch.utils.data
from torcheval.metrics import Mean
import math
import os
import logging


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


def score(y_pred, y_true, min_tpr: float = 0.80) -> float:
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

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(y_true - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * y_pred

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

    #     # Equivalent code that uses sklearn's roc_auc_score
    #     v_gt = abs(np.asarray(solution.values)-1)
    #     v_pred = np.array([1.0 - x for x in submission.values])
    #     max_fpr = abs(1-min_tpr)
    #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return (partial_auc)


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
    def __init__(self, path, logger):
        self.path = path
        self.logger = logger
        self.best_validation_loss = float('inf')
        os.makedirs(self.path, exist_ok=True)

    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.path, "best.pt"))

    def __call__(self, validation_loss, model):
        if validation_loss < self.best_validation_loss:
            self.save_model(model)
            self.logger.info(
                f"Saved model's weight improvement from {self.best_validation_loss} to {validation_loss}")
            self.best_validation_loss = validation_loss


class MixUp():
    def __init__(self, alpha=0.2):
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, batch):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        x, y = torch.utils.data.default_collate(batch)
        lam = self.beta_dist.sample()
        index = torch.randperm(x.size()[0])

        return lam * x + (1 - lam) * x[index, :], y, y[index], lam


def train_single_task(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs, alpha=0.2):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='results/training.log', encoding='utf-8', level=logging.INFO)

    train_history_epoch_loss = []
    val_history_epoch_loss = []

    save_best_model = SaveBestModel("checkpoint_resnet50_mix_up", logger=logger)
    early_stopping = EarlyStopping(tolerance=5)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_epoch_loss = Mean().to("cpu")
    val_epoch_loss = Mean().to("cpu")

    train_loss_epoch_value = val_loss_epoch_value = 0

    with tqdm(total=epochs) as epoch_p_bar:
        for epoch in range(epochs):
            train_batch_processed = 1

            model.train()
            for train_batch in train_dataloader:
                mix_x, y_a, y_b, lam = (train_batch[0].to(device=device, dtype=torch.float),
                                        train_batch[1].to(device=device, dtype=torch.float),
                                        train_batch[2].to(device=device, dtype=torch.float),
                                        train_batch[3].to(device=device, dtype=torch.float))

                train_batch_len = len(mix_x)

                optimizer.zero_grad(set_to_none=True)

                y_train_pred = model(mix_x).to(device=device)

                train_loss = (criterion(input=y_train_pred, target=y_a) * lam +
                              criterion(input=y_train_pred, target=y_b) * (1-lam))

                train_loss.backward()
                optimizer.step()

                train_epoch_loss.update(train_loss.detach().cpu() * train_batch_len)
                train_loss_epoch_value = train_epoch_loss.compute().item()
                metric_update = (f"[Train loss: {round(train_loss_epoch_value, 4)}] \
                - [Val_loss: {round(val_loss_epoch_value, 4)}]")

                epoch_p_bar.set_description(f"{metric_update} | Train batch processed: {train_batch_processed} \
                - {len(train_dataloader)} - {math.ceil(train_batch_processed / len(train_dataloader) * 100)}%")
                train_batch_processed += 1

            # Validation
            model.eval()
            with torch.no_grad():
                val_batch_processed = 1
                for val_batch in val_dataloader:
                    x_val, y_val = (val_batch[0].to(device=device, dtype=torch.float),
                                    val_batch[1].to(device=device, dtype=torch.float))

                    val_batch_len = len(x_val)

                    y_val_pred = model(x_val)

                    val_loss = criterion(input=y_val_pred, target=y_val)

                    val_epoch_loss.update(val_loss.detach().cpu() * val_batch_len)
                    val_loss_epoch_value = val_epoch_loss.compute().item()
                    metric_update = (f"[Train loss: {round(train_loss_epoch_value, 4)}] \
                    - [Val_loss: {round(val_loss_epoch_value, 4)}]")

                    epoch_p_bar.set_description(
                        f"{metric_update} | Val batch processed: {val_batch_processed} from {len(val_dataloader)} \
                        - {math.ceil(val_batch_processed / len(val_dataloader) * 100)}%")
                    val_batch_processed += 1

            #Clean metrics state at the end of the epoch
            train_epoch_loss.reset()
            val_epoch_loss.reset()
            #Scheduler step
            scheduler.step()

            train_history_epoch_loss.append(train_loss_epoch_value)
            val_history_epoch_loss.append(val_loss_epoch_value)

            save_best_model(validation_loss = val_loss_epoch_value, model=model)
            if early_stopping(validation_loss = val_loss_epoch_value):
                logger.info("Stopped early")
                break

            epoch_p_bar.update(1)

    return train_history_epoch_loss, val_history_epoch_loss
