---
title: "Quick and easy GPU & TPU acceleration for PyTorch with HuggingFace Accelerate"
date: "2022-01-07"
categories: 
  - "deep-learning"
  - "frameworks"
  - "geen-categorie"
tags: 
  - "acceleration"
  - "deep-learning"
  - "gpu"
  - "huggingface"
  - "machine-learning"
  - "tpu"
---

Deep learning benefits from Graphical Processing Units (GPUs) and Tensor Processing Units (TPUs) because of the way they handle the necessary computations during model training. GPU and TPU based acceleration can thus help you speed up your model training process greatly.

Unfortunately, accelerating your PyTorch model on a GPU or TPU has quite a bit of overhead in native PyTorch: you'll need to assign the data, the model, the optimizer, and so forth, to the `device` object that contains a reference to your accelerator. It's very easy to forget it just once, and then your model breaks.

In today's article, we're going to take a look at **HuggingFace Accelerate** - a PyTorch package that abstracts away the overhead and allows you to accelerate your neural network with only a few lines of Python code. In other words, it allows you to **quickly and easily accelerate your deep learning model with GPU and TPU**.

Let's take a look! :)

* * *

\[toc\]

* * *

## What is HuggingFace Accelerate?

If you're familiar to the machine learning world, it's likely that you have heard of HuggingFace already - because they are known for their [Transformers library](https://www.machinecurve.com/index.php/getting-started-with-huggingface-transformers/). HuggingFace itself is a company providing an AI community "building the future of AI".

And that's why they provide [a lot more libraries](https://github.com/huggingface) which can be very useful to you as a machine learning engineer!

In today's article, we're going to take a look at **quick and easy** **accelerating** for your **PyTorch deep learning model** using your **GPU or TPU.**

This can be accomplished with `accelerate`, a [HuggingFace package](https://github.com/huggingface/accelerate) that can be described in the following way:

> 🚀 A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
> 
> GitHub (n.d.)

Who doesn't want to benefit from speed when you have the hardware available?

Let's continue by looking at how it works :D

* * *

## How to install HuggingFace Accelerate?

Installing HuggingFace is very easy. Obviously, you will need to have a recent install of Python and PyTorch (the package was tested with Python 3.6+ and PyTorch 1.4.0+). Then, it's only the execution of a `pip` command:

```
pip install accelerate
```

* * *

## Easy GPU/TPU acceleration for PyTorch - Python example

Now that you have installed HuggingFace Accelerate, it's time to accelerate our PyTorch model 🤗

Obviously, a model is necessary if you want to accelerate it, so that is why we will use a model that we created before, [in another blog article](https://www.machinecurve.com/index.php/2021/01/26/creating-a-multilayer-perceptron-with-pytorch-and-lightning/). It's a simple Multilayer Perceptron that is trained for classification with the CIFAR-10 dataset, and you will find an explanation as to how it works when clicking the link.

Today, however, we will simply use it for acceleration with HuggingFace Accelerate. Here, you can find the code - which, as you can see, has no references to `cuda` whatsoever and hence runs on CPU by default:

```
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
      nn.ReLU(),
      nn.Linear(64, 32),
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

The first thing that you will need to do is ensuring that HuggingFace `accelerate` is imported. You can do this by adding the following to the imports:

```
from accelerate import Accelerator
```

Immediately afterwards, you then initialize the accelerator:

```
accelerator = Accelerator()
```

That's pretty much it when it comes to loading stuff, you can now immediately use it by accelerating the model (`mlp`), the optimizer (`optimizer`) and `DataLoader` (`trainloader`) - just before the training loop of your MLP:

```
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

  # Accelerate the model, optimizer and trainloader
  mlp, optimizer, trainloader = accelerator.prepare(mlp, optimizer, trainloader)
```

Now, the only thing you will need to do is changing the backward pass by the functionality provided by the accelerator, so that it is performed in an accelerated way:

```
      # Compute loss
      loss = loss_function(outputs, targets)
      
      # Perform backward pass
      accelerator.backward(loss)
```

That's it - here's the full code if you want to get started straight away :)

```
import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator

accelerator = Accelerator()

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
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

  # Accelerate the model, optimizer and trainloader
  mlp, optimizer, trainloader = accelerator.prepare(mlp, optimizer, trainloader)
  
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
      accelerator.backward(loss)
      
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

That's it!

You have accelerated your PyTorch model by letting it use your GPU or TPU when available!

If you have any questions, comments or suggestions, feel free to leave a message in the comments section below 💬 I will then try to answer you as quickly as possible. For now, thank you for reading MachineCurve today and happy engineering! 😎

* * *

## References

GitHub. (n.d.). _Huggingface/accelerate: 🚀 a simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision_. [https://github.com/huggingface/accelerate](https://github.com/huggingface/accelerate)
