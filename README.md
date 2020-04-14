# boilertorch
## Boilerplate code for PyTorch projects
boilertorch contains boilerplate / template code for PyTorch projects provided by the `TorchGadget` class

`TorchGadget` features:
- Complete built-in pipeline for classification tasks
- Customizable methods for other tasks
- Built-in training loop: system config checks, training, reporting to stdout and checkpoint saving and loading
- Modularized inference / prediction and evaluation
- Standardized checkpoint contents
  - Model state
  - Optimizer state
  - Scheduler state (if applicable)
  - Lists of loss and/or evaluation metric values over the training and/or development datasets across training epochs (if computed)
- System checks 
  - Warns user if running the model on 'cpu' (warning if `torch.device` is 'cpu')
  - Warns user if the model save directory is not empty

## Instructions:
You can focus on the fun part!
- Write your own PyTorch `nn.Module` model class and initialize the model parameters yourself
- Write your own `torch.utils.data.Dataset` class. (When instantiating blind test data without given target outputs, use dummy outputs instead.)
- Batch your data using `torch.utils.data.Dataloader` 
- Extend the `TorchGadget` class with custom code if your model is not a classification model

We take care of the rest!
- Instantiate a `TorchGadget` object
- Train your model using `TorchGadget.train`, loading a checkpoint if desired
- Generate predictions from your model using `TorchGadget.predict_set`

### Built-in classification pipeline
The built-in pipeline assumes your model is a binary or multi-class classification model that outputs softmax logits. It generates prediction by taking the maximum output logit and the built-in metric is accuracy.

Example:
```
from boilertorch import TorchGadget

train_loader = Dataloader(...)
dev_loader = Dataloader(...)
test_loader = Dataloader(...)

model = MyModel(...)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=...)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, ...)  # optional
checkpt_path = 'path/checkpt.pt'  # optional

gadget = TorchGagdet(model, optimizer, scheduler, checkpoint=checkpt_path)
gadget.train(criterion, train_loader, dev_loader, n_epochs=50, save_dir='./')
dev_accuracy = gadget.eval_set(dev_loader)  # evaluate accuracy over the dev set
predictions = gadget.predict_set(test_loader)  # predict labels for the test set

```


### Customizing model forward output
- The built-in pipeline assumes your model takes a single argument and returns a single output, like so:
```
outputs = self.model(x)
```
- If your model's forward function does not follow this signature, overload or edit the `TorchGagdet.get_outputs` method to put a batch from the Dataloader through your model's forward function.


### Customizing training criterion / loss function
- The built-in loss computation assumes the criterion takes outputs and target as the argument, like so:
```
loss = criterion(outputs, target)
```
Most loss functions already follow this signature e.g. `nn.CrossEntropyLoss`, `nn.MSELoss`, `nn.NLLLoss`, `nn.L1Loss`
- If your criterion does not follow this signature (e.g. `nn.CTCLoss`), overload or edit the `TorchGagdet.compute_loss` method to put outputs from your model and the targets into the criterion.


### Customizing model prediction and evaluation metric
- If your model is not a binary or multi-class classification model that outputs softmax logits, overload or edit the `TorchGagdet.get_predictions` method to put a batch from the Dataloader through your model and generate predictions. `Torchgadget` uses this for both evaluation and blind prediction. For blind prediction, `TorchGagdet.predict_set` will take care of concatenating the outputs of the batches for you if your `TorchGagdet.get_predictions` method returns the predictions as a `torch.Tensor`, `numpy.ndarray` or Python `list`.
- If your evaluation metric is not classification accuracy, overload or edit the `TorchGagdet.compute_metric` method to compute your evaluation metric with the outputs or predictions from your model and the targets.