---
title: "Using Dropout with PyTorch"
date: "2021-07-07"
categories: 
  - "deep-learning"
  - "frameworks"
tags: 
  - "deep-learning"
  - "dropout"
  - "machine-learning"
  - "neural-network"
  - "overfitting"
  - "pytorch"
---

The Dropout technique can be used for avoiding overfitting in your neural network. It has been around for some time and is widely available in a variety of neural network libraries. Let's take a look at how Dropout can be implemented with PyTorch.

In this article, you will learn...

- **How variance and overfitting are related.**
- **What Dropout is and how it works against overfitting.**
- **How Dropout can be implemented with PyTorch**.

Let's take a look! 😎

* * *

\[toc\]

* * *

## Variance and overfitting

In our article about the [trade-off between bias and variance](https://www.machinecurve.com/index.php/2020/11/02/machine-learning-error-bias-variance-and-irreducible-error-with-python/), it became clear that models can be high in _bias_ or high in _variance_. Preferably, there is a balance between both.

To summarize that article briefly, models high in bias are relatively rigid. Linear models are a good example - they assume that your input data has a linear pattern. Models high in variance, however, do not make such assumptions -- but they are sensitive to changes in your training data.

As you can imagine, striking a balance between rigidity and sensitivity.

Dropout is related to the fact that deep neural networks have high variance. As you know when you have dealt with neural networks for a while, such models are sensitive to overfitting - capturing noise in the data as if it is part of the real function that must be modeled.

* * *

## Dropout

In their paper [“Dropout: A Simple Way to Prevent Neural Networks from Overfitting”](http://jmlr.org/papers/v15/srivastava14a.html), Srivastava et al. (2014) describe Dropout, which is a technique that temporarily removes neurons from the neural network.

> With Dropout, the training process essentially drops out neurons in a neural network.
> 
> [What is Dropout? Reduce overfitting in your neural networks](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/)

When certain neurons are dropped, no data flows through them anymore. Dropout is modeled as [Bernoulli variables](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/#bernoulli-variables), which are either zero (0) or one (1). They can be configured with a variable, \[latex\]p\[/latex\], which illustrates the probability (between 0 and 1) with which neurons are dropped.

When neurons are dropped, they are not dropped permanently: instead, at every epoch (or even minibatch) the network randomly selects neurons that are dropped this time. Neurons that had been dropped before can be activated again during future iterations.

- For a more detailed explanation of Dropout, see our article [_What is Dropout? Reduce overfitting in your neural networks_](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/).

* * *

## Using Dropout with PyTorch: full example

Now that we understand what Dropout is, we can take a look at how Dropout can be implemented with the PyTorch framework. For this example, we are using a [basic example](https://www.machinecurve.com/index.php/2021/01/26/creating-a-multilayer-perceptron-with-pytorch-and-lightning/) that models a Multilayer Perceptron. We will be applying it to the MNIST dataset (but note that Convolutional Neural Networks are more applicable, generally speaking, for image datasets).

In the example, you'll see that:

- We import a variety of dependencies. These include `os` for Python operating system interfaces, `torch` representing PyTorch, and a variety of sub components, such as its neural networks library (`nn`), the `MNIST` dataset, the `DataLoader` for loading the data, and `transforms` for a Tensor transform.
- We define the `MLP` class, which is a PyTorch neural network module (`nn.Module`). Its constructor initializes the `nn.Module` super class and then initializes a `Sequential` network (i.e., a network where layers are stacked on top of each other). It begins by flattening the three-dimensional input (width, height, channels) into a one-dimensional input, then applies a `Linear` layer (MLP layer), followed by Dropout, Rectified Linear Unit. This is then repeated once more, before we end with a final `Linear` layer for the final multiclass prediction.
- The `forward` definition is a relatively standard PyTorch definition that must be included in a `nn.Module`: it ensures that the forward pass of the network (i.e., when the data is fed to the network), is performed by feeding input data `x` through the `layers` defined in the constructor.
- In the `main` check, a random seed is fixed, the dataset is loaded and prepared; the MLP, loss function and optimizer are initialized; then the model is trained. This is the classic PyTorch training loop: gradients are zeroed, a forward pass is performed, loss is computed and backpropagated through the network, and optimization is performed. Finally, after every iteration, statistics are printed.

```
import os
import torch
from torch import nn
from torchvision.datasets import MNIST
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
      nn.Linear(28 * 28 * 1, 64),      
      nn.Dropout(p=0.5),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.Dropout(p=0.5),
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
  dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
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

## References

PyTorch. (n.d.). _Dropout — PyTorch 1.9.0 documentation_. [https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

MachineCurve. (2019, December 17). _What is dropout? Reduce overfitting in your neural networks_. [https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/)
