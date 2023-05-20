################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
    model.eval()
    correct, total_samples = 0.,0.

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            batch_size = data_inputs.size(0)
            data_inputs = data_inputs.view(batch_size, -1)
            outputs = model(data_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += data_labels.size(0)
            correct += (predicted == data_labels).sum().item()

    accuracy = correct / total_samples
    #print(f"Accuracy of the model: {100.0 * accuracy:4.2f}%")
    return accuracy


def plots(loss_values):  # not working- fix!!
    # Training Loss Curve
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.title('Training Loss Curve')
    plt.show



# def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
#     """
#     Performs a full training cycle of MLP model.
#
#     Args:
#       hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
#       lr: Learning rate of the SGD to apply.
#       use_batch_norm: If True, adds batch normalization layer into the network.
#       batch_size: Minibatch size for the data loaders.
#       epochs: Number of training epochs to perform.
#       seed: Seed to use for reproducible results.
#       data_dir: Directory where to store/find the CIFAR10 dataset.
#     Returns:
#       model: An instance of 'MLP', the trained model that performed best on the validation set.
#       val_accuracies: A list of scalar floats, containing the accuracies of the model on the
#                       validation set per epoch (element 0 - performance after epoch 1)
#       test_accuracy: scalar float, average accuracy on the test dataset of the model that
#                      performed best on the validation.
#       logging_info: An arbitrary object containing logging information. This is for you to
#                     decide what to put in here.
#
#     TODO:
#     - Implement the training of the MLP model.
#     - Evaluate your model on the whole validation set each epoch.
#     - After finishing training, evaluate your model that performed best on the validation set,
#       on the whole test dataset.
#     - Integrate _all_ input arguments of this function in your training. You are allowed to add
#       additional input argument if you assign it a default value that represents the plain training
#       (e.g. '..., new_param=False')
#
#     Hint: you can save your best model by deepcopy-ing it.
#     """
#
#     # Set the random seeds for reproducibility
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     # Create TensorBoard logger
#     logging_dir = 'runs/our_experiment'
#     os.makedirs(logging_dir, exist_ok=True)
#     writer = SummaryWriter(logging_dir)
#     model_plotted = False
#
#
#     if torch.cuda.is_available():  # GPU operation have separate seed
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.determinstic = True
#         torch.backends.cudnn.benchmark = False
#
#     # Set default device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Loading the dataset
#     cifar10 = cifar10_utils.get_cifar10(data_dir)
#     cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
#                                                   return_numpy=False)
#
#     model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10, use_batch_norm=use_batch_norm).to(device)
#     loss_module = nn.CrossEntropyLoss()   # This function already performs the softmax operation internally as part of its computation.
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     best_model = None
#     train_accuracies = []
#     val_accuracies = []
#     loss_values =[]
#     best_val_accuracy = 0.0
#     model.train()
#     # Training loop including validation
#     for epoch in tqdm(range(epochs)):
#         epoch_loss = 0.0
#         for data_inputs, data_labels in cifar10_loader["train"]:
#             # Move input data to device
#             data_inputs = data_inputs.to(device)
#             data_labels = data_labels.to(device)   # in size [128] (as batch size)
#             batch_size = data_inputs.size(0)
#             data_inputs = data_inputs.view(batch_size, -1)  # reshape from [128, 3, 32, 32] to [128, 3072]
#             # Run the model on the input data
#             preds = model(data_inputs)    # in size [128,10] (128- batch size, 10- num of classes)
#             # Calculate the loss
#             loss = loss_module(preds, data_labels.long())
#             # Backpropagation
#             # First- we need to ensure that they are all zero.
#             optimizer.zero_grad()
#             # Perform backpropagation
#             loss.backward()
#
#             # Update the parameters
#             optimizer.step()
#             epoch_loss += loss.item()
#
#             writer.add_scalar('Loss', epoch_loss, global_step=epoch + 1)
#             writer.add_scalar('Validation Accuracy', val_accuracy, global_step=epoch + 1)
#
#         loss_values.append(epoch_loss)
#         #train_accuracy = evaluate_model(model, cifar10_loader["train"], num_classes=10)
#         #train_accuracies.append(train_accuracy)
#         val_accuracy = evaluate_model(model, cifar10_loader["validation"], num_classes=10)
#         val_accuracies.append(val_accuracy)
#
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             best_model = deepcopy(model)
#         print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Val Accuracy={100.0 * val_accuracy:.2f}%")
#     test_accuracy = evaluate_model(best_model, cifar10_loader["test"], num_classes=10)
#     print(f"The accuracy of the test set with the best model: {100.0 * test_accuracy:.2f}%")
#     logging_info = None  # Placeholder for any additional logging information you might want to save
#     # Plot the loss curve
#     # Add average loss to TensorBoard
#     writer.close()
#     logging_info = logging_dir
#     #plots(loss_values)
#     return model, val_accuracies, test_accuracy, logging_info

from torch.utils.tensorboard import SummaryWriter
import os
from copy import deepcopy

def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
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

    # Create TensorBoard logger
    logging_dir = 'runs/our_experiment'
    os.makedirs(logging_dir, exist_ok=True)
    writer = SummaryWriter(logging_dir)
    model_plotted = False

    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10, use_batch_norm=use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()   # This function already performs the softmax operation internally as part of its computation.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    best_model = None
    train_accuracies = []
    val_accuracies = []
    loss_values = []
    best_val_accuracy = 0.0
    model.train()

    # Training loop including validation
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0

        for data_inputs, data_labels in cifar10_loader["train"]:
            # Move input data to device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)   # in size [128] (as batch size)
            batch_size = data_inputs.size(0)
            data_inputs = data_inputs.view(batch_size, -1)  # reshape from [128, 3, 32, 32] to [128, 3072]

            # Run the model on the input data
            preds = model(data_inputs)    # in size [128,10] (128- batch size, 10- num of classes)

            # Calculate the loss
            loss = loss_module(preds, data_labels.long())

            # Backpropagation
            # First, we need to ensure that gradients are all zero.
            optimizer.zero_grad()

            # Perform backpropagation
            loss.backward()

            # Update the parameters
            optimizer.step()
            epoch_loss += loss.item()

        writer.add_scalar('Loss', epoch_loss, global_step=epoch + 1)

        val_accuracy = evaluate_model(model, cifar10_loader["validation"], num_classes=10)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Val Accuracy={100.0 * val_accuracy:.2f}%")

    test_accuracy = evaluate_model(best_model, cifar10_loader["test"], num_classes=10)
    print(f"The accuracy of the test set with the best model: {100.0 * test_accuracy:.2f}%")

    writer.close()
    logging_info = logging_dir

    return model, val_accuracies, test_accuracy, logging_info



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
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



