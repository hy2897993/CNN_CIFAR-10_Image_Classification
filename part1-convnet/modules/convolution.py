"""
2d Convolution Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape
        h_out = (H + 2*self.padding - self.kernel_size)//self.stride + 1
        w_out = (W + 2*self.padding - self.kernel_size)//self.stride + 1
        out = np.zeros((N,self.out_channels,h_out, w_out))
        x_padded = np.pad(x, ((0,),(0,),(self.padding,),(self.padding,)))

        for n in range(N):
            for c in range(self.out_channels):
                for h in range(h_out):
                    for w in range(w_out):
                        out[n,c,h,w] = np.sum(x_padded[n,:,(h*self.stride):(h*self.stride+self.kernel_size),(w*self.stride):(w*self.stride+self.kernel_size)]  * self.weight[c,:] )+ self.bias[c]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape
        h_out = (H + 2*self.padding - self.kernel_size)//self.stride + 1
        w_out = (W + 2*self.padding - self.kernel_size)//self.stride + 1
        x_padded = np.pad(x, ((0,),(0,),(self.padding,),(self.padding,)))

        
        self.dx = np.zeros(x.shape)
        
        self.dw = np.zeros(self.weight.shape)
        
        self.db = np.zeros(self.bias.shape)
        pad_dx = np.pad(self.dx, ((0,),(0,),(self.padding,),(self.padding,)))

        for n in range(N):
            for c in range(self.out_channels):
                self.db[c] += np.sum(dout[n,c])
                for h in range(h_out):
                    for w in range(w_out):
                        self.dw[c] += x_padded[n,:,h*self.stride:h*self.stride+self.kernel_size,\
                            w*self.stride:w*self.stride+self.kernel_size] * dout[n,c,h,w]
                        pad_dx[n,:,h*self.stride:h*self.stride+self.kernel_size,\
                            w*self.stride:w*self.stride+self.kernel_size] += self.weight[c] * dout[n,c,h,w]
        self.dx = pad_dx[:,:,self.padding:H+self.padding,self.padding:W + self.padding]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
