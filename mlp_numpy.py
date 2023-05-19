################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################

"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logsistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        self.linear1 = LinearModule(n_inputs, n_hidden)
        self.linear2 = LinearModule(n_hidden, n_classes)
        self.relu_fn = RELUModule()
        self.soft_max = SoftMaxModule()

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:s
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """
        out = self.linear1(x)
        out = self.relu_fn(out)
        out = self.linear2(out)
        out = self.soft(out)
        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss with respect to the network output

        TODO:
        Implement backward pass of the network.
        """
        dout = self.soft.backward(dout)
        dout = self.linear2.backward(dout)
        dout = self.relu_fn.backward(dout)
        dout = self.linear1.backward(dout)
        return dout

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        for attribute_name in dir(self):   # iterate over the modules of the network
            attribute = getattr(self, attribute_name)
            if callable(getattr(attribute, 'clear_cache', None)):
                attribute.clear_cache()