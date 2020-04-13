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
        self.train_loss = None
        self.train_metric = None
        self.dev_loss = None
        self.dev_metric = None

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def __repr__(self):
        str_joins = []
        str_joins.append(f"Device: {self.device}")
        str_joins.append(str(self.model))
        str_joins.append(str(self.optimizer) if self.optimizer else "No optimizer provided.")
        str_joins.append(str(self.scheduler) if self.scheduler else "No scheduler provided.")
        str_joins.append(f"Epoch: {self.epoch}")
        if self.train_loss:
            str_joins.append(f"Train loss: {self.train_loss}")
        if self.train_metric:
            str_joins.append(f"Train metric: {self.train_metric}")
        if self.dev_loss:
            str_joins.append(f"Development loss: {self.dev_loss}")
        if self.dev_metric:
            str_joins.append(f"Development metric: {self.dev_metric}")
        return '\n'.join(str_joins)

    def get_batch_outputs(self, batch):
        """
        CUSTOMIZABLE FUNCTION
        Takes a batch and runs it through the model's forward method to get outputs.
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        # Unpack batch
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Compute loss
        outputs = self.model(x)

        # Clean up
        del x
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return outputs, y

    def get_batch_predictions(self, batch, **kwargs):
        """
        CUSTOMIZABLE FUNCTION
        Takes a batch and generates predictions from the model e.g. class labels for classification models
        Overload this function appropriately with your Dataset class's output and model generation function signature
        """
        # Unpack batch
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Generate predictions
        outputs = self.model(x)
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        # Clean up
        del x, outputs
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return pred_labels, y

    def compute_batch_loss(self, batch, criterion):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average loss over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        outputs, target = self.get_batch_outputs(batch)
        loss = criterion(outputs, target)

        # Clean up
        del outputs, target
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return loss

    def compute_batch_metric(self, batch, **kwargs):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average evaluation metric over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward or inference function
        signature
        """
        predictions, target = self.get_batch_predictions(batch)
        metric = self._accuracy(predictions, target)  # can be changed to any evaluation metric, with optional kwargs

        # Clean up
        del predictions, target
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return metric

    def _accuracy(self, pred_labels, labels):
        """Mean accuracy over predicted and true labels"""
        batch_size = len(labels)
        accuracy = torch.sum(torch.eq(pred_labels, labels)).item() / batch_size
        return accuracy

    def train(self, train_loader, n_epochs, criterion, eval_train=False, dev_loader=None,
              save_dir='./', save_freq=1, report_freq=0, **kwargs):
        """
        Boiler plate training procedure, optionally saves the model checkpoint after every epoch
        :param train_loader: training set dataloader
        :param dev_loader: development set dataloader
        :param n_epochs: the epoch number after which to terminate training
        :param criterion: the loss criterion
        :param save_dir: the save directory
        :param report_freq: report training approx report_freq times per epoch
        """
        # Check config
        assert self.optimizer is not None, "Optimizer required for training. Set TorchGadget.optimizer"
        if self.scheduler and not eval_train and not dev_loader:
            print("Warning: Use of scheduler without evaluating either the training or validatation sets per epoch")
            print("If scheduler is dynamic, it only compares training loss over one reporting cycle (may be unstable).")

        save_dir = self.check_save_dir(save_dir)
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'

        self.model.to(self.device)
        criterion.to(self.device)


        # Prepare to begin training
        batch_group_size = int(len(train_loader) / report_freq) if report_freq else 0  # report every batch_group_size

        print(f"Training set batches: {len(train_loader)}\tBatch size: {train_loader.batch_size}.")
        print(f"Development set batches: {len(dev_loader)}\tBatch size: {dev_loader.batch_size}." if dev_loader \
              else "No development set provided")
        print(f"Beginning training at {datetime.now()}")
        if self.epoch == 0:
            with open(save_dir + "results.txt", mode='a') as f:
                header = "epoch"
                if eval_train:
                    header = header + ",train_loss,train_metric"
                if dev_loader:
                    header = header + ",dev_loss,dev_metric"
                f.write(header + "\n")

        if self.epoch == 0 or (eval_train and (self.train_loss is None or self.train_metric is None)):
            self.train_loss = []
            self.train_metric = []
        if self.epoch == 0 or (dev_loader and (self.dev_loss is None or self.dev_metric is None)):
            self.dev_loss = []
            self.dev_metric = []


        # Train
        self.model.train()
        for epoch in range(self.epoch + 1, n_epochs + 1):
            # Train over epoch
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
                    print(f'Epoch {epoch}, Batch {batch_num}\tTrain loss: {avg_loss / batch_group_size:.4f}\t{datetime.now()}')
                    avg_loss = 0.0

                # Cleanup
                del batch
                torch.cuda.empty_cache()

            # Evaluate epoch
            with open(save_dir + "results.txt", mode='a') as f:
                line = str(epoch)
                if eval_train:  # evaluate over training set
                    epoch_train_loss = self.eval_set(train_loader, self.compute_loss, criterion=criterion)
                    epoch_train_metric = self.eval_set(train_loader, self.compute_metric, **kwargs)
                    self.train_loss.append(epoch_train_loss)
                    self.train_metric.append(epoch_train_metric)
                    line = line + f",{epoch_train_loss},{epoch_train_metric}"
                if dev_loader:  # evaluate over development set
                    epoch_dev_loss = self.eval_set(dev_loader, self.compute_loss, criterion=criterion)
                    epoch_dev_metric = self.eval_set(dev_loader, self.compute_metric, **kwargs)
                    self.dev_loss.append(epoch_dev_loss)
                    self.dev_metric.append(epoch_dev_metric)
                    line = line + f",{epoch_dev_loss},{epoch_dev_metric}"
                f.write(line + "\n")

            # Step scheduler
            if self.scheduler:
                if dev_loader:
                    self.try_sched_step(epoch_dev_loss)
                elif eval_train:
                    self.try_sched_step(epoch_train_loss)
                else:
                    self.try_sched_step(avg_loss)

            # Save checkpoint
            if save_freq and epoch % save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': getattr(self.scheduler, 'state_dict', lambda: None)(),
                    'train_loss': self.train_loss,
                    'train_metric': self.train_metric,
                    'dev_loss': self.dev_loss,
                    'dev_metric': self.dev_metric
                }
                torch.save(checkpoint, save_dir + f"checkpoint_{epoch}_{epoch_dev_metric:.4f}.pth")

            # Print epoch log
            epoch_log = f'Epoch {epoch} complete.'
            if eval_train:
                epoch_log = epoch_log + f"\tTrain loss: {epoch_train_loss:.4f}\tTrain metric: {epoch_train_metric:.4f}"
            if dev_loader:
                epoch_log = epoch_log + f"\tDev loss: {epoch_dev_loss:.4f}\tDev metric: {epoch_dev_metric:.4f}"
            print(f'{epoch_log}\t{datetime.now()}')

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
        :return: Concatenated predictions of the same type as returned by self.get_batch_predictions
        """
        self.model.eval()
        accum = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                predictions_batch, _ = self.get_batch_predictions(batch, **kwargs)
                # predictions_batch = outputs.detach().to('cpu')
                accum.extend(predictions_batch)

                # Clean up
                del batch
                torch.cuda.empty_cache()

        self.model.train()

        if isinstance(predictions_batch, list):
            predictions_set = accum
        elif isinstance(predictions_batch, torch.Tensor):
            predictions_set = torch.cat(accum)
        elif isinstance(predictions_batch, np.ndarray):
            predictions_set = np.concatenate(accum)
        else:
            raise NotImplementedError("Unsupported output type: ", type(predictions_batch))

        del predictions_batch
        torch.cuda.empty_cache()

        return predictions_set

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
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except (AttributeError, KeyError):
                    pass
                self.epoch = checkpoint['epoch']
                if 'dev_loss' in checkpoint:
                    self.dev_loss = checkpoint['dev_loss']
                if 'dev_metric' in checkpoint:
                    self.dev_metric = checkpoint['dev_metric']
                done = True
            except FileNotFoundError:
                print(f"Provided checkpoint path {checkpoint_path} not found. Path must include the filename itself.")
                checkpoint_path = input("Provide a new path, or [init] to use randomized weights: ")


    def try_sched_step(self, metrics):
        """
        Steps the scheduler
        First tries to step with the metric in case the scheduler is dynamic, then falls back to a step without args.
        """
        try:
            self.scheduler.step(metrics=metrics)
        except TypeError:
            self.scheduler.step()

    def check_save_dir(self, save_dir):
        """Checks that the provided save directory exists and warns the user if it is not empty"""
        done = False
        while not done:
            if save_dir == 'exit':
                print("Exiting")
                sys.exit(0)

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
                    f"Save directory {save_dir} is not empty. Do you want to overwrite it? [y/n] ")
                if use_save_dir == 'n':
                    save_dir = input("Enter new save directory (or [exit] to exit): ")
                elif use_save_dir == 'y':
                    sure = input(f"Are you sure you want to overwrite the directory {save_dir}? [y/n] ")
                    if sure == 'y':
                        import shutil
                        assert save_dir != '~/' and save_dir != '~'
                        shutil.rmtree(save_dir)
                        os.makedirs(save_dir)
                        done = True
                    elif sure != 'n':
                        print("Please enter one of [y/n]")
                else:
                    print("Please enter one of [y/n]")
            else:
                done = True

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