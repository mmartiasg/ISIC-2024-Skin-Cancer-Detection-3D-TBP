import torch
from tqdm.auto import tqdm
import numpy as np
from tqdm.auto import tqdm
import numpy as np
import torch.utils.data
from torcheval.metrics import MulticlassAccuracy, Mean


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


def mix_up_data(x, y, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def train_single_task(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs, alpha=0.2):
    train_history_epoch_loss = []
    val_history_epoch_loss = []

    train_epoch_loss = Mean().to("cpu")
    val_epoch_loss = Mean().to("cpu")

    with tqdm(total=epochs) as epoch_p_bar:
        for epoch in range(epochs):

            model.train()
            for train_batch in train_dataloader:
                x = train_batch[0].to(device=device)
                y = train_batch[1].to(device=device)

                mix_x, y_a, y_b, lam = mix_up_data(x, y, alpha=alpha)

                mix_x, y_a, y_b = mix_x.to(device=device, dtype=torch.float), y_a.to(device=device, dtype=torch.float), y_b.to(device=device, dtype=torch.float)

                train_batch_len = len(x)

                y_train_pred = model(mix_x).to(device=device)

                train_loss = criterion(input=y_train_pred, target=y_a) * lam + criterion(input=y_train_pred, target=y_b) * (1-lam)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_epoch_loss.update(train_loss.detach().cpu() * train_batch_len)
                model.train()


            model.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    x_val = val_batch[0].to(device=device)
                    y_val = val_batch[1].to(device=device)

                    x_val, y_val = x_val.to(device=device), y_val.to(device=device)
                    val_batch_len = len(x_val)

                    y_val_pred = model(x_val)

                    val_loss = criterion(input=y_val_pred, target=y_val)

                    val_epoch_loss.update(val_loss.detach().cpu() * val_batch_len)
            model.train()

            train_history_epoch_loss.append(train_epoch_loss.compute().item())
            val_history_epoch_loss.append(val_epoch_loss.compute().item())

            epoch_p_bar.set_description(f"[loss: {round(train_history_epoch_loss[-1], 4)} - val_loss: {round(val_history_epoch_loss[-1], 4)}]")

            #Clean metrics state at the end of the epoch
            train_epoch_loss.reset()
            val_epoch_loss.reset()


            epoch_p_bar.update(1)


    return train_history_epoch_loss, val_history_epoch_loss
