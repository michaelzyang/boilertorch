import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

class TorchGadget():
    def __init__(self, model, optimizer=None, scheduler=None, checkpoint='', device=None):
        """Instantiate TorchGadget with torch objects and optionally a checkpoint filename"""
        self.device = device if device else self.get_device()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def __repr__(self):
        str_joins = []
        str_joins.append(f"Device: {self.device}")
        str_joins.append(str(self.model))
        str_joins.append(str(self.optimizer) if self.optimizer else "No optimizer provided.")
        str_joins.append(str(self.scheduler) if self.scheduler else "No scheduler provided.")
        str_joins.append(f"Epoch: {self.epoch}")
        return '\n'.join(str_joins)

    def compute_loss(self, batch, criterion):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average loss over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        # Unpack batch
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Compute loss
        outputs = self.model(x)
        loss = criterion(outputs, y)

        # Clean up
        del x, y, outputs
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return loss

    def compute_metric(self, batch, **kwargs):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average evaluation metric over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward or inference function
        signature
        """
        # Unpack batch
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Compute accuracy
        outputs = self.model(x)  # can be changed to any generative method, with optional kwargs
        # outputs = outputs.detach().to('cpu')
        metric = self._accuracy(outputs, y)  # can be changed to any evaluation metric, with optional kwargs

        # Clean up
        del x, y, outputs
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return metric


    def _accuracy(self, outputs, labels):
        """Mean accuracy over a batch, given labels and logit outputs"""
        batch_size = len(labels)
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        accuracy = torch.sum(torch.eq(pred_labels, labels)).item() / batch_size

        # Clean up
        del pred_labels

        return accuracy

    def compute_predictions(self, batch, **kwargs):
        """
        CUSTOMIZABLE FUNCTION
        Generates predictions for evaluation over a batch of data.
        Overload this function appropriately with your Dataset class's output and model forward or inference function
        signature
        :return: default supported output types are torch.Tensor, np.ndarray or list
        """
        # Unpack batch
        x = batch[0]

        # Compute prediction
        outputs = self.model(x)  # can be changed to any generative method, with optional kwargs
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        # Clean up
        del x, outputs
        # no need to del batch or pred_labels (deleted in the calling function)
        torch.cuda.empty_cache()

        return pred_labels

    def train(self, train_loader, dev_loader, n_epochs, criterion, save_dir='./', save_freq=1,
              report_freq=0, **kwargs):
        """
        Boiler plate training procedure, optionally saves the model checkpoint after every epoch
        :param train_loader: training set dataloader
        :param dev_loader: development set dataloader
        :param n_epochs: the epoch number after which to terminate training
        :param criterion: the loss criterion
        :param save_dir: the save directory
        :param report_freq: report training approx report_freq times per epoch
        """
        # Setup
        assert self.optimizer is not None, "Optimizer required for training. Set TorchGadget.optimizer"

        save_dir = self.check_save_dir(save_dir)
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'

        self.model.to(self.device)
        criterion.to(self.device)

        batch_group_size = int(
            len(train_loader) / report_freq) if report_freq else 0  # print status every batch_group_size

        print(f"Beginning training at {datetime.now()}")
        if self.epoch == 0:
            with open(save_dir + "results.txt", mode='a') as f:
                f.write("epoch,dev_loss,dev_metric\n")

        # Train epochs
        self.model.train()
        for epoch in range(self.epoch + 1, n_epochs + 1):
            avg_loss = 0.0  # Accumulate loss over subsets of batches for reporting
            for i, batch in enumerate(train_loader):
                batch_num = i + 1
                self.optimizer.zero_grad()

                self.model.to(self.device)
                loss = self.compute_loss(batch, criterion)
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                # Accumulate loss for reporting
                avg_loss += loss.item()
                if batch_group_size and (batch_num) % batch_group_size == 0:
                    print(
                        f'Epoch {epoch}, Batch {batch_num}\tTrain loss: {avg_loss / batch_group_size:.4f}\t{datetime.now()}')
                    avg_loss = 0.0

                # Cleanup
                del batch
                torch.cuda.empty_cache()

            # Evaluate epoch
            dev_loss = self.eval_set(dev_loader, self.compute_loss, criterion=criterion)
            dev_metric = self.eval_set(dev_loader, self.compute_metric, **kwargs)
            with open(save_dir + "results.txt", mode='a') as f:
                f.write(f"{epoch},{dev_loss},{dev_metric}\n")

            if self.scheduler:
                self.scheduler.step(dev_loss)

            if save_freq and epoch % save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'dev_loss': dev_loss,
                    'dev_metric': dev_metric
                }
                torch.save(checkpoint, save_dir + f"checkpoint_{epoch}_{dev_metric:.4f}.pth")

            print(f'Epoch {epoch} complete.\tDev loss: {dev_loss:.4f}\tDev metric: {dev_metric:.4f}\t{datetime.now()}')
        print(f"Finished training at {datetime.now()}")

    def eval_set(self, data_loader, compute_fn, **kwargs):
        """
        BOILERPLATE FUNCTION
        Evaluates the average loss or evaluation metric of the model on a given dataset
        :param data_loader: A dataloader for the data over which to evaluate
        :param compute_fn: Either the compute_loss or compute_metric method
        :param kwargs: either The criterion for compute_loss or other kwargs for compute_metric
        :return: The average loss or metric per sentence
        """
        self.model.eval()
        accum = 0.0
        batch_count = 0.0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                num_data_points = batch[0].shape[0]
                if i == 0:
                    batch_size = num_data_points  # assumes batch first ordering

                # Accumulate
                accum += compute_fn(batch, **kwargs)
                batch_count += num_data_points / batch_size  # last batch may be smaller than batch_size

                # Clean up
                del batch
                torch.cuda.empty_cache()

        self.model.train()

        return accum / batch_count

    def predict_set(self, data_loader, **kwargs):
        """
        BOILERPLATE FUNCTION
        Generates the predictions of the model on a given dataset
        :param data_loader: A dataloader for the data over which to evaluate
        :param kwargs: any kwargs required for prediction
        :return: Concatenated predictions of the same type as returned by self.compute_predictions
        """
        self.model.eval()
        accum = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                pred_labels = self.compute_predictions(batch, **kwargs)
                accum.extend(pred_labels)

                # Clean up
                del batch
                torch.cuda.empty_cache()

        self.model.train()

        if isinstance(pred_labels, list):
            predictions = accum
        elif isinstance(pred_labels, torch.Tensor):
            predictions = torch.cat(accum)
        elif isinstance(pred_labels, np.ndarray):
            predictions = np.concatenate(accum)
        else:
            raise NotImplementedError("Unsupported output type: ", type(pred_labels))

        del pred_labels
        torch.cuda.empty_cache()

        return predictions

    def load_checkpoint(self, checkpoint_path=''):
        """Loads a checkpoint for the model, epoch number and optimizer if provided"""
        # Try loading checkpoint and keep asking for valid checkpoint paths upon failure.
        done = False
        while not done:
            if checkpoint_path == 'init':
                self.epoch = 0
                done = True
            try:
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                done = True
            except FileNotFoundError:
                print(f"Provided checkpoint path {checkpoint_path} not found. Path must include the filename itself.")
                checkpoint_path = input("Provide a new path, or [init] to use randomized weights: ")


    def check_save_dir(self, save_dir):
        """Checks that the provided save directory exists and warns the user if it is not empty"""
        done = False
        while not done:
            #  Make sure save_dir exists
            if not os.path.exists(save_dir):
                print(f"Save directory {save_dir} does not exist.")
                mkdir_save = input(
                    "Do you wish to create the directory [m], enter a different directory [n], or exit [e]? ")
                if mkdir_save == 'm':
                    os.makedirs(save_dir)
                    done = True
                    continue
                elif mkdir_save == 'n':
                    save_dir = input("Enter new save directory: ")
                    continue
                elif mkdir_save == 'e':
                    sys.exit()
                else:
                    print("Please enter one of [m/n/e]")
                    continue

            #  Ensure user knows if save_dir is not empty
            if os.listdir(save_dir):
                use_save_dir = input(
                    f"Save directory {save_dir} is not empty. Are you sure you want to continue? [y/n] ")
                if use_save_dir == 'n':
                    save_dir = input("Enter new save directory (or [exit] to exit): ")
                elif use_save_dir == 'y':
                    done = True
                else:
                    print("Please enter one of [y/n]")

        return save_dir

    def get_device(self):
        """Gets the preferred torch.device and warns the user if it is cpu"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            use_cpu = input("Torch device is set to 'cpu'. Are you sure you want to continue? [y/n] ")
            while use_cpu not in ('y', 'n'):
                print("Please input y or n")
                use_cpu = input("Torch device is set to 'cpu'. Are you sure you want to continue? [y/n] ")
            if use_cpu == 'n':
                sys.exit()
        return device