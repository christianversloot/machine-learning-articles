---
title: "Batch Normalization with PyTorch"
date: "2021-03-29"
categories:
  - "buffer"
  - "deep-learning"
  - "frameworks"
tags:
  - "batch-normalization"
  - "covariance-shift"
  - "deep-learning"
  - "neural-network"
  - "neural-networks"
  - "pytorch"
---

One of the key elements that is considered to be a good practice in a neural network is a technique called Batch Normalization. Allowing your neural network to use normalized inputs across all the layers, the technique can ensure that models converge faster and hence require less computational resources to be trained.

In a different tutorial, we showed how you can implement [Batch Normalization with TensorFlow and Keras](https://www.machinecurve.com/index.php/2020/01/15/how-to-use-batch-normalization-with-keras/). This tutorial focuses on **PyTorch** instead. After reading it, you will understand:

- **What Batch Normalization does at a high level, with references to more detailed articles.**
- **The differences between `nn.BatchNorm1d` and `nn.BatchNorm2d` in PyTorch.**
- **How you can implement Batch Normalization with PyTorch.**

It also includes a test run to see whether it can really perform better compared to not applying it.

Let's take a look! 🚀

* * *

\[toc\]

* * *

## Full code example: Batch Normalization with PyTorch

```python
import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
  
if __name__ == '__main__':
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Prepare CIFAR-10 dataset
  dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
  
  # Initialize the MLP
  mlp = MLP()
  
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  
  # Run the training loop
  for epoch in range(0, 5): # 5 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    
    # Set current loss value
    current_loss = 0.0
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):
      
      # Get inputs
      inputs, targets = data
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)
      
      # Compute loss
      loss = loss_function(outputs, targets)
      
      # Perform backward pass
      loss.backward()
      
      # Perform optimization
      optimizer.step()
      
      # Print statistics
      current_loss += loss.item()
      if i % 500 == 499:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
          current_loss = 0.0

  # Process is complete.
  print('Training process has finished.')
```

* * *

## What is Batch Normalization?

Training a neural network is performed according to the high-level supervised machine learning process. A batch of data is fed through the model, after which its predictions are compared with the actual or _ground truth_ values for the inputs.

The difference leads to what is known as a loss value, which can be used for subsequent error backpropagation and model optimization.

Optimizing a model involves slightly adapting the weights of the trainable layers in your model. All is good so far. However, now suppose that you have the following scenario:

- You feed a model with a batch of low-dimensional data that has a mean of 0.25 and a standard deviation of 1.2, and you adapt your model.
- Your second batch has a mean of 13.2 and an 33.9 standard deviation.
- Your third goes back to 0.35 and 1.9, respectively.

You can imagine that given your model's weights, it will be relatively poor in handling the second batch - and by consequence, the weights change significantly. By consequence, your model will also be _worse_ than it can be when processing the third batch, simply because it has adapted for the significantly deviating scenario.

And although it can learn to reverse to the more generic process over time, you can see that with relative instability in your dataset (which can even happen within relatively normalized datasets, due to such effects happening in downstream layers), model optimization will oscillate quite heavily. And this is bad, because it slows down the training process.

**Batch Normalization** is a normalization technique that can be applied at the layer level. Put simply, it normalizes "the inputs to each layer to a learnt representation likely close to \[latex\](\\mu = 0.0, \\sigma = 1.0)\[/latex\]. By consequence, all the layer inputs are normalized, and significant outliers are less likely to impact the training process in a negative way. And if they do, their impact will be much lower than without using Batch Normalization.

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization, and in some cases eliminates the need for Dropout. Applied to a stateof-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.82% top-5 test error, exceeding the accuracy of human raters.
>
> The abstract from the [Batch Normalization paper](http://proceedings.mlr.press/v37/ioffe15.html) by Ioffe & Szegedy (2015)

* * *

## BatchNormalization with PyTorch

If you wish to understand Batch Normalization in more detail, I recommend reading our [dedicated article about Batch Normalization](https://www.machinecurve.com/index.php/2020/01/14/what-is-batch-normalization-for-training-neural-networks/). Here, you will continue implementing Batch Normalization with the PyTorch library for deep learning. This involves a few steps:

1. Taking a look at the differences between `nn.BatchNorm2d` and `nn.BatchNorm1d`.
2. Writing your neural network and constructing your Batch Normalization-impacted training loop.
3. Consolidating everything in the full code.

### Differences between BatchNorm2d and BatchNorm1d

First of all, the differences between two-dimensional and one-dimensional Batch Normalization in PyTorch.

1. Two-dimensional Batch Normalization is made available by `[nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)`.
2. For one-dimensional Batch Normalization, you can use `[nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)`.

One-dimensional Batch Normalization is defined as follows on the PyTorch website:

> Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputs with optional additional channel dimension) (...)
>
> PyTorch (n.d.)

...this is how two-dimensional Batch Normalization is described:

> Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) (…)
>
> PyTorch (n.d.)

Let's summarize:

- One-dimensional BatchNormalization (`nn.BatchNorm1d`) applies Batch Normalization over a 2D or 3D input (a _batch_ of _1D_ inputs with a possible _channel_ dimension).
- Two-dimensional BatchNormalization (`nn.BatchNorm2d`) applies it over a 4D input (a _batch_ of _2D_ inputs with a possible _channel_ dimension).

#### 4D, 3D and 2D inputs to BatchNormalization

Now, what is a "4D input"? PyTorch describes it as follows: \[latex\](N, C, H, W)\[/latex\]

- Here, \[latex\]N\[/latex\] stands for the number of samples in a batch.
- \[latex\]C\[/latex\] represents the number of channels.
- \[latex\]H\[/latex\] represents height and \[latex\]W\[/latex\] width.

In other words, a 4D input to a `nn.BatchNorm2d` layer represents a set of \[latex\]N\[/latex\] objects that each have a height and a width, always a number of channels >= 1. What comes to mind when reading that?

Indeed, images do.

A "2D or 3D input" goes as follows: \[latex\](N, C, L)\[/latex\] (here, the C is optional).

`nn.BatchNorm1d` represents lower-dimensional inputs: a number of inputs, possibly a number of channels and a content per object. These are regular, one-dimensional arrays, like the ones produced by [Dense layers](https://www.machinecurve.com/index.php/2019/07/27/how-to-create-a-basic-mlp-classifier-with-the-keras-sequential-api/) in a neural network.

Okay: we now know that we must apply `nn.BatchNorm2d` to layers that handle images. Primarily, these are [Convolutional layers](https://www.machinecurve.com/index.php/2018/12/07/convolutional-neural-networks-and-their-components-for-computer-vision/), which slide over images in order to generate a more abstract representation of them. `nn.BatchNorm1d` can be used with Dense layers that are stacked on top of the Convolutional ones in order to generate classifications.

#### Where to use BatchNormalization in your neural network

Now that we know what _type_ of Batch Normalization must be applied to each type of layer in a neural network, we can wonder about the _where_ - i.e., where to apply Batch Normalization in our neural network.

Here's the advice of some Deep Learning experts:

> Andrew Ng says that batch normalization should be applied immediately before the non-linearity of the current layer. The authors of the BN paper said that as well, but now according to François Chollet on the keras thread, the BN paper authors use BN after the activation layer. On the other hand, there are some benchmarks (…) that show BN performing better after the activation layers.
>
> StackOverflow (n.d.)

There is thus no clear answer to this question. You will have to try experimentally what works best.

### Writing the neural network + training loop

Okay, we now know the following things...

- What Batch Normalization does at a high level.
- Which types of Batch Normalization we need for what type of layer.
- Where to apply Batch Normalization in your neural network.

Time to talk about the core of this tutorial: implementing Batch Normalization in your PyTorch based neural network. Applying Batch Normalization to a PyTorch based [neural network](https://www.machinecurve.com/index.php/2021/01/26/creating-a-multilayer-perceptron-with-pytorch-and-lightning/) involves just three steps:

1. Stating the imports.
2. Defining the `nn.Module`, which includes the application of Batch Normalization.
3. Writing the training loop.

Create a file - e.g. `batchnorm.py` - and open it in your code editor. Also make sure that you have Python, PyTorch and `torchvision` installed onto your system (or available within your Python environment). Let's go!

#### Stating the imports

Firstly, we're going to state our imports.

- We're going to need `os` based definitions for downloading the dataset properly.
- All `torch` based imports are required for PyTorch: `torch` itself, the `nn` (a.k.a. neural network) module and the `DataLoader` for loading the dataset we're going to use in today's neural network.
- From `torchvision`, we load the `CIFAR10` dataset - as well as some `transforms` (primarily image normalization) that we will apply on the dataset before training the neural network.

```python
import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
```

#### Defining the nn.Module, with Batch Normalization

Next up is defining the `nn.Module`. Indeed, we're not using `Conv` layers today - which will likely improve your neural network. Instead, we're immediately flattening the 32x32x3 input, then further processing it into a 10-class outcome (because CIFAR10 has 10 classes).

As you can see, we're applying `BatchNorm1d` here because we use densely-connected/fully connected (a.k.a. `Linear`) layers. Note that the number of inputs to the BatchNorm layer must equal the number of _outputs_ of the `Linear` layer.

It clearly shows how Batch Normalization must be applied with PyTorch.

```python
class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
```

#### Writing the training loop

Next up is writing the training loop. We're not going to cover it to a great extent here, because already wrote about it in our dedicated article about getting started with a first PyTorch model:

- [Getting started with PyTorch](https://www.machinecurve.com/index.php/2021/01/13/getting-started-with-pytorch/)

However, to summarize briefly what happens, here you go:

- First, we set the seed vector of our random number generator to a fixed number. This ensures that any differences are due to the stochastic nature of the number generation process, and not due to pseudorandomness of the number generator itself.
- We then prepare the CIFAR-10 dataset, initialize the MLP and define the loss function and optimizer.
- This is followed by iterating over the epochs, where we set current loss to 0.0 and start iterating over the data loader. We set the gradients to zero, perform the forward pass, compute the loss, and perform the backwards pass followed by optimization. Indeed this is what happens in the supervised ML process.
- We print statistics per mini batch fed forward through the model.

```python
if __name__ == '__main__':
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Prepare CIFAR-10 dataset
  dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
  
  # Initialize the MLP
  mlp = MLP()
  
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  
  # Run the training loop
  for epoch in range(0, 5): # 5 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    
    # Set current loss value
    current_loss = 0.0
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):
      
      # Get inputs
      inputs, targets = data
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)
      
      # Compute loss
      loss = loss_function(outputs, targets)
      
      # Perform backward pass
      loss.backward()
      
      # Perform optimization
      optimizer.step()
      
      # Print statistics
      current_loss += loss.item()
      if i % 500 == 499:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
          current_loss = 0.0

  # Process is complete.
  print('Training process has finished.')
```

### Full code

I can imagine why you want to get started immediately. It's always more fun to play around, isn't it? :) You can find the full code for this tutorial at the top of this page.

* * *

## Results

These are the results after training our MLP for 5 epochs on the CIFAR-10 dataset, _with_ Batch Normalization:

```shell
Starting epoch 5
Loss after mini-batch   500: 1.573
Loss after mini-batch  1000: 1.570
Loss after mini-batch  1500: 1.594
Loss after mini-batch  2000: 1.568
Loss after mini-batch  2500: 1.609
Loss after mini-batch  3000: 1.573
Loss after mini-batch  3500: 1.570
Loss after mini-batch  4000: 1.571
Loss after mini-batch  4500: 1.571
Loss after mini-batch  5000: 1.584
```

The same, but then _without_ Batch Normalization:

```shell
Starting epoch 5
Loss after mini-batch   500: 1.650
Loss after mini-batch  1000: 1.656
Loss after mini-batch  1500: 1.668
Loss after mini-batch  2000: 1.651
Loss after mini-batch  2500: 1.664
Loss after mini-batch  3000: 1.649
Loss after mini-batch  3500: 1.647
Loss after mini-batch  4000: 1.648
Loss after mini-batch  4500: 1.620
Loss after mini-batch  5000: 1.648
```

Clearly, but unsurprisingly, the Batch Normalization based model performs better.

* * *

## Summary

In this tutorial, you have read about implementing Batch Normalization with the PyTorch library for deep learning. Batch Normalization, which was already proposed in 2015, is a technique for normalizing the inputs to each layer within a neural network. This can ensure that your neural network trains faster and hence converges earlier, saving you valuable computational resources.

After reading it, you now understand...

- **What Batch Normalization does at a high level, with references to more detailed articles.**
- **The differences between `nn.BatchNorm1d` and `nn.BatchNorm2d` in PyTorch.**
- **How you can implement Batch Normalization with PyTorch.**

Great! Your next step may be to enhance your training process even further. Take a look at our article about [K-fold Cross Validation](https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/) for doing so.

I hope that it was useful for your learning process! Please feel free to share what you have learned in the comments section 💬 I’d love to hear from you. Please do the same if you have any questions or other remarks.

Thank you for reading MachineCurve today and happy engineering! 😎

* * *

## References

PyTorch. (n.d.). _BatchNorm1d — PyTorch 1.8.0 documentation_. [https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

PyTorch. (n.d.). _BatchNorm2d — PyTorch 1.8.0 documentation_. [https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

StackOverflow. (n.d.). _Where to apply batch normalization on standard CNNs_. Stack Overflow. [https://stackoverflow.com/questions/47143521/where-to-apply-batch-normalization-on-standard-cnns](https://stackoverflow.com/questions/47143521/where-to-apply-batch-normalization-on-standard-cnns)

Ioffe, S., & Szegedy, C. (2015, June). [Batch normalization: Accelerating deep network training by reducing internal covariate shift.](http://proceedings.mlr.press/v37/ioffe15.html) In _International conference on machine learning_ (pp. 448-456). PMLR.
