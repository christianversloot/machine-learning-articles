---
title: "How to save and load a PyTorch model?"
date: "2021-02-03"
categories:
  - "buffer"
  - "deep-learning"
  - "frameworks"
tags:
  - "deep-learning"
  - "load-model"
  - "machine-learning"
  - "pytorch"
  - "save-model"
---

You don't train deep learning models without using them later. Instead, you want to save them, in order to load them later - allowing you to perform inference activities.

In this tutorial, we're going to take a look at **saving and loading your models created with PyTorch**. PyTorch is one of the leading frameworks for deep learning these days and is widely used in the deep learning industry. After reading it, you will understand...

- How you can use `torch.save` for saving your PyTorch model.
- How you can load the model by initializing the skeleton and loading the state.

Let's take a look! 😎

* * *

\[toc\]

* * *

## Saving a PyTorch model

Suppose that you have created a PyTorch model, say a simple Multilayer Perceptron, like this.

```python
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
      nn.Conv2d(1, 5, kernel_size=3),
      nn.Flatten(),
      nn.Linear(26 * 26 * 5, 300),
      nn.ReLU(),
      nn.Linear(300, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
```

You can then define a [training loop](https://www.machinecurve.com/index.php/2021/01/26/creating-a-multilayer-perceptron-with-pytorch-and-lightning/#defining-the-training-loop) in order to train the model, in this case with the MNIST dataset. Note that we don't repeat creating the training loop here - click the link to see how this can be done.

After training, it is possible that you have found a model that is useful in the real world.

In other words, a well-performing model that must be saved.

And saving a deep learning model with PyTorch is actually really easy - the only thing that you have to do is call `torch.save`, like this:

```python
# Saving the model
save_path = './mlp.pth'
torch.save(mlp.state_dict(), save_path)
```

Here, you define a path to a PyTorch (`.pth`) file, and save the state of the model (i.e. the weights) to that particular file. Note that `mlp` here is the initialization of the neural network, i.e. we executed `mlp = MLP()` during the construction of your training loop. `mlp` is thus any object instantiated based on your `nn.Module` extending neural network class.

When you run your model next time, the state gets saved to a file called `./mlp.pth`.

* * *

## Loading a saved PyTorch model

...but things don't end there. When you saved a PyTorch model, you likely want to load it at a different location.

For inference, for example, meaning that you will use it in a deployment setting for generating predictions.

Loading the model is however really easy and involves the following steps:

1. Initializing the model skeleton.
2. Loading the model state from a file defined at a particular path.
3. Setting the state of your model to the state just loaded.
4. Evaluating the model.

```python
# Loading the model
mlp = MLP()
mlp.load_state_dict(torch.load(save_path))
mlp.eval()
```

That's it!

* * *

## Recap

After training a deep learning model with PyTorch, it's time to use it. This requires you to save your model. In this tutorial, we covered how you can **save and load your PyTorch models** using `torch.save` and `torch.load`.

I hope that you have learned something from this article, despite it being really short - and shorter than you're used to when reading this website! Still, there's no point in writing a lot of text when the important things can be said with only few words, is there? :)

If you have questions, please feel free to reach out in the comments section below 💬

Thank you for reading MachineCurve today and happy engineering! 😎

* * *

## References

PyTorch. (n.d.). [https://pytorch.org](https://pytorch.org/)
