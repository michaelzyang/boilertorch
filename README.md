# boilertorch
## Boilerplate code for PyTorch projects
boilertorch contains boilerplate / template code for PyTorch projects wrapped in a TorchGadget class

TorchGadget features:
- Modularized training loop: system config checks, reporting to stdout and checkpoint saving and loading
- Modularized inference
- Standardized checkpoint contents

## Instructions:
You can focus on the fun part!
- Write your own PyTorch `nn.Module` model class and initialize the model parameters yourself
- Write your own `torch.utils.data.Dataset` class. (When instantiating blind test data without given target outputs, use dummy outputs instead.)
- Batch your data using `torch.utils.data.Dataloader` 
- Extend the `TorchGadget` class with custom code if your model is not a binary or multi-class classification model

We take care of the rest!
- Instantiate a `TorchGadget` object
- Train your model using `TorchGadget.train`, loading a checkpoint if desired
- Generate predictions from your model using `TorchGadget.predict_set`


### Built-in pipeline
- The built-in pipeline assumes your model takes a single argument and returns a single output, like so:
```
outputs = self.model(x)  # Signature 1
```
- The built-in loss computation assumes the criterion takes outputs and target as the argument, like so:
```
loss = criterion(outputs, target)  # Signature 2
```
Most loss functions have this signature e.g. `nn.CrossEntropyLoss`, `nn.MSELoss`, `nn.NLLLoss`, `nn.L1Loss`
- The built-in pipeline assumes your model is a binary or multi-class classification model that outputs softmax logits. It generates prediction by taking the maximum output logit and the built-in metric is accuracy.

### Customizing model forward output and training criterion / loss function
- If your model's forward function does not follow Signature 1 above in the built-in pipeline, overload or edit the `get_batch_outputs` method to put a batch from the Dataloader through your model's forward function. (It is likely that you will not have to do this.)
- If your criterion does not follow Signature 2 above in the built-in pipeline, overload or edit the `compute_batch_loss` method to put outputs from your model and the targets into the criterion.

### Customizing model prediction and evaluation metric
- If your model is not a binary or multi-class classification model that outputs softmax logits, overload or edit the `get_batch_predictions` method to put a batch from the Dataloader through your model and generate predictions. `Torchgadget` uses this for both evaluation and blind prediction. For blind prediction, `TorchGagdet.predict_set` will take care of concatenating the outputs of the batches for you if your `get_batch_predictions` method returns the predictions as a `torch.Tensor`, `numpy.ndarray` or Python `list`.
- If your evaluation metric is not classification accuracy, overload or edit the `compute_batch_metric` method to compute your evaluation metric with the outputs or predictions from your model and the targets.