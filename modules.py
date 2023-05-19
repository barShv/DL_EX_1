################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
# Ophir Gal and Bar Shvarzman
################################################################################

"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """
        if input_layer:
            # meaning this is the first layer after the input and we just randomly initialize weights

            self.weight = np.random.randn(out_features, in_features) # this is N x M
        else:
            # in hidden layers, we need to use Kaiming Initilization which divides all WEIGHTS by an stdv factor, assuming W~N(o,2/in_features)
            std = np.sqrt(2 / in_features)  # Kaiming initialization standard deviation (this is good for ReLu activation)
            #self.weight = np.random.randn(in_features,out_features) * std
            ####### (I am not sure if we should also make sure that the means are zero at first.. ). This would be how it's done:
            self.weight = np.random.normal(loc=0.0, scale=std, size=(out_features, in_features))
            # self.weight is (128,3072)
            #print(self.weight.shape)

        self.bias = np.zeros(out_features) #this is 1XN
        self.grads = {'weight': np.zeros_like(self.weight), #Initialize two gradients, one for Weights and one for Biases (gradients are same dimensions)
              'bias': np.zeros_like(self.bias)}
        #print(self.bias.shape)

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # The basic linear transformation Y = XW(T) + B
        out = np.dot(x, self.weight.T) + self.bias

        # Store intermediate variables for backward pass, X is the input of the previous layer
        self.input = x
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        # Compute the gradients with respect to the weights and biases
        self.grads['weight'] = np.dot(self.input.T, dout) # x^T*dout, The closed form expression for is: ∂L/∂W =( ∂L/∂Y))· X (or flip matrixes order with T)
        self.grads['bias'] = np.sum(dout, axis=0) # dout*I, where I is a column of 1's.

        # Compute the gradients with respect to the input of the module
        print(f"dout size {dout.shape}")
        dx = np.dot(dout, self.weight.T) #∂L/∂x =(∂L/∂Y))· W^⊤
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.input = None


class RELUModule(object):
    """
    RELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        out = np.maximum(0, x)  # Apply ReLU function
        #intermediate parameters
        self.input = x
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        dx = np.where(self.input > 0, dout, 0)  # Gradient is 1*dout when x>0, else 0
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.input = None


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # The max trick : exp(xi−b)/(∑exp(xj−b)), where b=max(Xi). With this choice, overflow due to exp is impossible.
        # The largest number exponentiated after shifting is 0.
        # x size-(SXC), output size-(SXC)
        b = x.max(axis=1, keepdims=True)  # maximum value per row
        y = np.exp(x - b)
        out = y / y.sum(axis=1, keepdims=True)  # normalize each row
        # Intermediate parameters
        self.input = x
        self.output = out
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        dx = dout * self.output * (1 - self.output)
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.input = None
        self.output = None


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """
        print(f"probabilities{x.shape}")
        print(f"data_labels{y.shape}")
        out = -np.mean(np.sum(y * np.log(x), axis=1)) # axis 1 -> by rows
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """
        s = x.shape[0]  # Get the batch size
        # Compute the gradient (T in theory is y here, Y is SXC (1 HOT-encoder), X is SXC, that why the element wise division works.)
        dx = -(1/s)*(y / x)
        return dx


#For checking
#
# linear = LinearModule(in_features=3, out_features=2)
#
# # Generate sample input data
# x = np.random.randn(4, 3)
# print('x:',x)
#
# # Perform forward pass
# output = linear.forward(x)
# print("Output:")
# print(output)
#
# # Generate sample gradients of the loss function
# dout = np.random.randn(4, 2)
# print("Dout:")
# print(dout)
#
# # Perform backward pass
# grads = linear.backward(dout)
# print("Gradients:")
# print(grads)
#
# # Clear the cache
# linear.clear_cache()
# print(linear.input)