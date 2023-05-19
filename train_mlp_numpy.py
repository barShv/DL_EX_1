################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        accuracy 

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """
    correct, total_samples = 0.,0.

    for data_inputs, data_labels in data_loader:
        probabilities = model.forward(data_inputs)
        predicted_labels = np.argmax(probabilities, axis=1)
        total_samples += data_labels.size(0)
        correct += np.sum(predicted_labels == data_labels)

    accuracy = correct / total_samples
    return accuracy


def convert_to_one_hot(y, num_classes):
    """
    Converts target labels to one-hot encoding.

    Args:
      y: target labels (shape: batch_size,)
      num_classes: number of classes

    Returns:
      one_hot: one-hot encoded labels (shape: batch_size X num_classes (SxC))
    """
    batch_size = len(y)
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), y] = 1
    return one_hot


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()
    best_model = None
    val_accuracies = []
    loss_values =[]
    best_val_accuracy = 0.0
    # TODO: Training loop including validation
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in cifar10_loader["train"]:  # data_labels in size [128] (as batch size)
            data_inputs = data_inputs.reshape(data_inputs.shape[0], -1)  # reshape from [128, 3, 32, 32] to [128, 3072]
            probabilities = model.forward(data_inputs)  # the output after the soft max, in size [128,10] (128- batch size, 10- num of classes)
            labels_one_hot = convert_to_one_hot(data_labels, 10)
            loss = loss_module.forward(probabilities, labels_one_hot)
            model.clear_cache()
            dout = loss_module.backward(probabilities, labels_one_hot)
            model.backward(dout)  # ask ofir!!!
            # Update the parameters ???????
            epoch_loss += loss.item()
        loss_values.append(epoch_loss)
        val_accuracy = evaluate_model(model, cifar10_loader["validation"], num_classes=10)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)
        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Val Accuracy={100.0 * val_accuracy:.2f}%")
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader["test"], num_classes=10)
    print(f"The accuracy of the test set with the best model: {100.0 * test_accuracy:.2f}%")
    # TODO: Add any information you might want to save for plotting
    logging_info = None  # Placeholder for any additional logging information you might want to save
    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    