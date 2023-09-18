---
title: "Getting started with PyTorch"
date: "2021-01-13"
categories:
  - "frameworks"
tags:
  - "deep-learning"
  - "getting-started"
  - "introduction"
  - "machine-learning"
  - "pytorch"
---

When you want to build a deep learning model these days, there are two machine learning libraries that you must consider. The first is [TensorFlow](https://www.machinecurve.com/index.php/mastering-keras/), about which we have written a lot on this website already. TensorFlow, having been created by Google and released to the public in 2015, has been the leading library for years. The second one is **PyTorch**, which was released by Facebook in 2016. Long running behind, both frameworks are now on par with each other, and are both used very frequently.

In this article, we will take a look at **getting started with PyTorch**. We will focus on simplicity of both our explanations and the code that we write. For this reason, we have chosen to work with [PyTorch Lightning](https://www.pytorchlightning.ai/) in the PyTorch articles on this website. Being a way to structure native PyTorch code, it helps boost reusability while saving a lot of overhead. In other words: you'll have the freedom of native PyTorch, while having the benefits of neat and clean code.

After reading this tutorial, you will have the answer to the question _"How to get started with PyTorch?"_. More specifically, you will...

- Know what steps you'll have to take in order to get started.
- Understand what PyTorch Lightning is and how it improves classic PyTorch.
- See how functionality in Lightning is organized in a `LightningModule` and how it works.
- Be able to set up PyTorch Lightning yourself.
- Have created your first PyTorch Lightning model.

* * *

**Update 20/Jan/2021:** Added `pl.seed_everything(42)` and `deterministic = True` to the code examples to ensure that pseudo-random number generator initialization happens with the same value, and use deterministic algorithms where available.

* * *

\[toc\]

* * *

## Quick start: 3 steps to get started with PyTorch Lightning

If you want to get started with PyTorch, **follow these 3 starting steps** to get started straight away! If you want to understand getting started with PyTorch in more detail, make sure to read the full tutorial. Here are the steps:

1. Ensure that Python, PyTorch and PyTorch Lightning are installed through `conda install pytorch-lightning -c conda-forge` and `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`.
2. Make sure that you understand what a `LightningModule` is, how it works and why it improves the model creation process over classic PyTorch.
3. Copy and paste the following example code into your editor and run it with Python.

```python
import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

class MNISTNetwork(pl.LightningModule):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )
    self.ce = nn.CrossEntropyLoss()
    
  def forward(self, x):
    return self.layers(x)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y)
    self.log('train_loss', loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer
  
  
if __name__ == '__main__':
  dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
  pl.seed_everything(42)
  neuralnetwork = MNISTNetwork()
  trainer = pl.Trainer(auto_scale_batch_size='power',gpus=1,deterministic=True)
  trainer.fit(neuralnetwork, DataLoader(dataset))
```

* * *

## What is PyTorch Lightning?

Today, when you want to create a deep learning model, you can choose **[PyTorch](https://pytorch.org/)** as the library of your choice. This library, which was released in September 2016 by Facebook, has become one of the two leading deep learning libraries. It is used by many researchers varying from academia to engineering, and is updated frequently.

[![](images/image-5-1024x487.png)](https://www.machinecurve.com/wp-content/uploads/2021/01/image-5.png)

The website of PyTorch Lightning

Native PyTorch models can be a bit disorganized, to say it nicely. They are essentially **long Python files** with all the elements you need, but **without any order**. For example, you'll have to…

- Declare the models and their structure.
- Define and load the dataset that you are using.
- Initialize optimizers and defining your custom training loop.

With **[PyTorch Lightning](https://www.pytorchlightning.ai/)**, this is no longer the case. It is a layer on top of native [PyTorch](https://pytorch.org/) and is hence compatible with all your original code - which can in fact be re-organized into Lightning code, to improve reusability. This is what makes Lightning different:

> Lightning makes coding complex networks simple.
>
> PyTorch Lightning (2021)

### Benefits of PyTorch Lightning over classic PyTorch

If we take a look at the benefits in more detail, we get to the following four:

1. **The same code, but then organized.**
2. **Trainer automates parts of the training process.**
3. **No **`.cuda()`** or **`.to()`** calls.**
4. **Built-in parallelism.**

Let's explore each in more detail now.

#### Benefit 1: The same code, but then organized

The _first benefit_ of using PyTorch Lightning is that **you'll have the same, PyTorch-compatible code, but then organized**. In fact, it "is just plain PyTorch" (PyTorch Lightning, 2021). Let's take a look at this example, which comes from the [Lightning website](https://www.pytorchlightning.ai/), and slightly adapted. We can see that the code is composed of a few segments that are all interrelated:

- The `models` segment specifies the neural network's encoder and decoder segments using the `torch.nn` APIs.
- Under `download data`, we download the MNIST dataset, and apply a transform to [normalize the data](https://www.machinecurve.com/index.php/2020/11/19/how-to-normalize-or-standardize-a-dataset-in-python/).
- We then generate a [train/test split](https://www.machinecurve.com/index.php/2020/11/16/how-to-easily-create-a-train-test-split-for-your-machine-learning-model/) of 55.000/5.000 images and load the data with `DataLoaders`.
- We specify an `optimizer`; the [Adam](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/) one in this case.
- Finally, we specify a custom training loop.

```python
# models
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(
    nn.Linear(28 * 28, 64), nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
)
encoder.cuda(0)
decoder.cuda(0)

# download data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)

# train (55,000 images), val split (5,000 images)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

# The dataloaders handle shuffling, batching, etc...
mnist_train = DataLoader(mnist_train, batch_size=64)
mnist_val = DataLoader(mnist_val, batch_size=64)

# optimizer
params = [encoder.parameters(), decoder.parameters()]
optimizer = torch.optim.Adam(params, lr=1e-3)

# TRAIN LOOP
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    for train_batch in mnist_train:
        x, y = train_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z = encoder(x)
        x_hat = decoder(z)
        loss = F.mse_loss(x_hat, x)
        print("train loss: ", loss.item())
```

And this is a simple model. You can imagine that when your model grows (and it does, because you'll have to write custom data loading and transformation segments; specify more layers; perhaps use custom loss functions and such), it'll become very difficult to see how things interrelate.

One of the key benefits of PyTorch Lightning is _that it organizes your code into a `LightningModule`._ We will cover this Lightning Module later in this article, and you will see that things are much more organized there!

#### Benefit 2: Trainer automates parts of the training process

In classic PyTorch, in the training loop, you have to write a lot of custom code, including...

- Instructing the model to get into training mode, enabling gradients to flow.
- Looping over the data loaders for training, validation and testing data; thus performing training, validation and testing activities.
- Computing loss for a batch, performing backprop, and applying the results with the optimizer.
- Defining device parallelism.

With PyTorch Lightning, this is no longer necessary either. The second benefit is that it **comes with a** **`Trainer` object** **that automates all the steps mentioned above, without forbidding control.**

> Once you’ve organized your PyTorch code into a LightningModule, the Trainer automates everything else.
>
> PyTorch Lightning (n.d.)

Yes: the `Trainer` automates training mode and gradient flow, automates the training loop, performs optimization, and allows you to tell PyTorch easily on what devices it must run and with what strategy.

No: this does not come at the cost of forfeiting control over your training process. Rather, while `Trainer` objects allow you to abstract away much of the training process, they allow you to customizer whatever part of the training process you want to customize. This allows you to get started quickly, while being able to configure the training process to your needs for when models are more complex.

#### Benefit 3: No .cuda() or .to() calls

This one's a bit more difficult, but the third benefit of PyTorch Lightning is **that you don't need to provide manual `.cuda()` and `.to()` calls**.

In order to understand what this means, you must realize that data processing on a GPU happens differently compared to processing on a CPU. GPU-based processing requires you to convert Tensors (i.e. the representations of data used within both TensorFlow and PyTorch) into CUDA objects; this is performed with `.cuda()`. Using `.to()`, you can also convert Tensors into different formats and across devices.

An example from the PyTorch docs is provided below (PyTorch, n.d.). In this example, three Tensors are created and possibly manipulated. The first Tensor is directly allocated to the first CUDA available device, i.e. a GPU. The second is first created on CPU and then transferred to the same GPU with `.cuda()`. The third is also first created on CPU and then transferred to a GPU, but then an explicitly defined one, using `.to(device=cuda)`.

```python
cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.tensor([1., 2.], device=cuda)

    # transfers a tensor from CPU to GPU 1
    b = torch.tensor([1., 2.]).cuda()
    # a.device and b.device are device(type='cuda', index=1)

    # You can also use ``Tensor.to`` to transfer a tensor:
    b2 = torch.tensor([1., 2.]).to(device=cuda)
```

While this gives you full control over the deployment of your model, it also comes at a cost: getting a correct configuration can be difficult. What's more, in many cases, your training setting is static over time - it's unlikely that you have 1.000 GPUs at your disposal at one time, and 3 at another time. This is why manual configuration of your CUDA devices and Tensor creation is an overhead at best and can be inefficient at worst.

PyTorch Lightning overcomes this issue by fully automating the `.cuda()`/`.to()` calls depending on the configuration provided in your `Trainer` object. You simply don't have to use them anymore in most of your code. Isn't that cool!

#### Benefit 4: Built-in parallelism

In classic PyTorch, when you want to train your model in a parallel setting (i.e. training on multiple GPUs), you had to build this into your code manually.

The fourth and final key benefit of PyTorch Lightning is that **Lightning takes care of parallelism when training your model, through the `Trainer` object.**

Indeed, adding parallelism is as simple as specifying e.g. the GPUs that you want to train your model on in the `Trainer` object (PyTorch Lightning, n.d.):

```python
Trainer(gpus=[0, 1])
```

And that's it - PyTorch Lightning takes care of the rest!

#### All benefits together

Let's sum a few things together now.

PyTorch Lightning improves classic PyTorch in the following ways:

1. **The same code, but then organized.**
2. **Trainer automates parts of the training process.**
3. **No `.cuda()` or `.to()` calls.**
4. **Built-in parallelism.**

But even then, you still have full control, and can override any automated choices made by Lightning. And what's more, it runs native PyTorch under the hood.

That's why we'll use Lightning in our PyTorch oriented tutorials as the library of choice.

* * *

## Introducing the LightningModule

Okay, so now we know why PyTorch Lightning improves PyTorch and that it can be used for constructing PyTorch models. Let's now take a look at the _what_, i.e. the `LightningModule` with which we'll work during the construction of our PyTorch models.

> A `LightningModule` is a [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) but with added functionality
>
> PyTorch Lightning (n.d.)

Here, the `torch.nn.Module` is the base class for all PyTorch based neural networks. In other words, a `LightningModule` is a layer on top of the basic way in which neural networks are constructed with PyTorch. It allows us to achieve the benefits that were outlined above, and in particular the benefit related to the organization of your machine learning model.

Each `LightningModule` is composed of six subsegments:

1. **Initialization segment**, or `__init__`. Essentially being the constructor of the `LightningModule` based class, it allows you to define the computations that must be used globally. For example, in this segment, you can specify the layers of your network and possibly how they are stacked together.
2. **Forward segment**, or `forward`. All inference data flows through `forward` and it therefore allows you to customize what happens during inference. Primarily though, this should be the generation of the prediction.
3. **Training segment**, or `training_step`. Here, you can specify the forward pass through the model during training, and the computation of loss. Upon returning the loss, PyTorch Lightning ensures that (1) the actual forward pass happens, that (2) errors with respect to loss are backpropagated, and that (3) the model is optimized with the optimizer of choice.
4. **Configure optimizers** through `configure_optimizers`. In this definition, you can specify the optimizer that you want to use.
5. **Validation segment** _(optional)_, or `validation_step`. Equal to the training segment, it is used for validating your model during the training process. Having a separate _validation step_ segment allows you to define a different validation approach, if necessary.
6. **Testing segment** _(optional)_, or `test_step`. Once again equal to the training segment, but then for evaluation purposes. It is not called during the training process, but rather when `.test()` is called on the `Trainer` object.

* * *

![](images/pexels-photo-1114690-1024x684.jpeg)

Getting started with PyTorch can be done at the speed of lightning - hence the name of the library.

* * *

## Setting up PyTorch Lightning

PyTorch Lightning can be installed really easily:

- **With PIP:** `pip install pytorch-lightning`
- **With Conda:** `conda install pytorch-lightning -c conda-forge`

That's all you need to get started with PyTorch Lightning!

If you are still missing packages after installation, also try the following:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

* * *

## Your first PyTorch model with Lightning

Now that we know both PyTorch Lightning and its `LightningModule`, it's time to show how you can build a neural network with PyTorch. Today, for introductory purposes, we will be creating a simple neural network that is capable of classifying the MNIST dataset. Building a neural network with PyTorch involves these five steps:

1. **Creating the LightningModule.**
2. **Defining the forward pass for inference.**
3. **Defining the training step.**
4. **Configuring the optimizers.**
5. **Setting the operational aspects.**

Let's now take a look at each individual one in more detail. Open up a Terminal and write some code.

### Creating the LightningModule

The first step involves specifying all the imports and creating the class that implements the `LightningModule` class.

With respect to the imports, we can say that we import the default modules. We will need `os` for dataset related activities. We use `torch` and its lower-level imports for PyTorch related aspects:

- The `nn` import defines building blocks for our neural network.
- The `DataLoader` can be used for loading the dataset into the model when training.
- From `torchvision`, we import both the `MNIST` dataset and `transforms`, the latter of which will be used for transforming the dataset into proper Tensor format later.
- Finally, we import PyTorch Lightning as `pl`.

Once this is completed, we can create the `LightningModule`. In fact, we create a class - called `MNISTNetwork` that implements the `LightningModule` class and hence has to implement many of its functions as well. The first definition that we implement is `__init__`, or the constructor function if you are familiar with object-oriented programming. Here, we:

- Also initialize the super class i.e. the instantiation of `pl.LightningModule` using `super().__init__()`.
- Define the neural network: using `nn.Sequential`, we can add our neural layers on top of each other. In this network, we're going to use three `Linear` layers that have ReLU activation functions and one final `Linear` layer.

```python
import os
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class MNISTNetwork(pl.LightningModule):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )
    self.ce = nn.CrossEntropyLoss()
```

### Defining the forward step for inference

The second step is to define the `forward` step that is used during inference. In other words, if a sample is passed into the model, you define here what should happen.

In our case, that is a pass of the input sample through our layers, and the output is returned.

```python
  def forward(self, x):
    return self.layers(x)
```

### Defining the training step

The third step is to define the training step, by means of the `training_step` definition. This definition accepts `batch` and `batch_idx` variables, where `batch` represents the items that are to be processed during this training step.

We first decompose the batch into `x` and `y` values, which contain the inputs and targets, respectively.

```python
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y)
    self.log('train_loss', loss)
    return loss
```

### Configuring the optimizers

We can then configure the optimizer. In this case, we use the Adam optimizer - which is a very common optimizer - and return it in our `configure_optimizers` definition. We set the default learning rate to \[latex\]10^-4\[/latex\] and let it use the model's parameters.

```python
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer
```

### Setting the operational aspects

That's it for creating the model. However, that's also only what we've got so far. We must add a few more things: loading and preparing the dataset, initializing the neural network, initializing the `Trainer` object (recall that it's a PyTorch Lightning feature that helps us automate aspects of the training process), and finally fitting the data.

We wrap all these aspects in `if __name__ == '__main__':`:

- Into `dataset`, we assign the `MNIST` dataset, which we download when it's not on our system and Transform into Tensor format using `transform=transforms.ToTensor()`.
- We use `pl.seed_everything(42)` to set a random seed for our pseudo-random number generator. This ensures full reproducibility regardless of pseudo-random number initialization (PyTorch Lightning, n.d.).
- We initialize the `MNISTNetwork` so that we can use our neural network.
- We initialize the PyTorch Lightning `Trainer` and instruct it to automatically scale batch size based on the hardware characteristics of our system. In addition, we instruct it to use a GPU device for training. If you don't have a dedicated GPU, you might use the CPU for training instead. In that case, simply remove `gpus=1`. Finally, we set `deterministic=True` to ensure reproducibility of the model (PyTorch LIghtning, n.d.).
- Finally, we apply `.fit(..)` and fit the `dataset` to the `neuralnetwork` by means of a `DataLoader`.

```python
if __name__ == '__main__':
  dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
  neuralnetwork= MNISTNetwork()

  trainer = pl.Trainer(auto_scale_batch_size='power',gpus=1,deterministic=True)
  trainer.fit(neuralnetwork, DataLoader(dataset))
```

### Full model code

Here's the full model code, for those who want to copy it and get started immediately.

```python
import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

class MNISTNetwork(pl.LightningModule):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )
    self.ce = nn.CrossEntropyLoss()
    
  def forward(self, x):
    return self.layers(x)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y)
    self.log('train_loss', loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer
  
  
if __name__ == '__main__':
  dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
  pl.seed_everything(42)
  neuralnetwork = MNISTNetwork()
  trainer = pl.Trainer(auto_scale_batch_size='power',gpus=1,deterministic=True)
  trainer.fit(neuralnetwork, DataLoader(dataset))
```

* * *

## Summary

PyTorch is one of the leading frameworks for machine learning these days, besides TensorFlow. In this article, we have started with PyTorch and showed you how you can get started too. First of all, we noticed that there are layers on top of PyTorch that can make your life easier as a PyTorch developer. We saw that with PyTorch Lightning, you don't have to worry about the organization of your code, parallelism of the training process, GPU deployment of your Tensors. In fact, many parts of the training process are automated away.

We then saw that a PyTorch Lightning module is called a `LightningModule` and that it consists of a few common building blocks that make it work. With the `__init__` definition, you can initialize the module, e.g. specifying the layers of your neural network. `Forward` can be used for specifying what should happen upon inference, i.e. when new samples are passed through the model. The `training_step`, `testing_step` and `validation_step` definitions describe what happens during the training, testing or validation steps, respectively. Finally, with `configure_optimizers`, you can choose what optimizer must be used for training the neural network and how it must be configured.

In an example implementation of a PyTorch model, we looked at how to construct a neural network using PyTorch in a step-by-step fashion. We saw that it's quite easy to do so once you understand the basics of neural networks and the way in which LightningModules are constructed. In fact, with our neural network, a classifier can be trained that is capable of classifying the MNIST dataset.

[Ask a question](https://www.machinecurve.com/index.php/add-machine-learning-question/)

I hope that this tutorial was useful! If you learned something, please feel free to leave a comment in the comments section 💬 Please do the same if you have questions, or leave a question through the **Ask Questions** button on the right.

Thank you for reading MachineCurve today and happy engineering! 😎

* * *

## References

PyTorch Lightning. (2021, January 12). [https://www.pytorchlightning.ai/](https://www.pytorchlightning.ai/)

PyTorch Lightning. (n.d.). _LightningModule — PyTorch lightning 1.1.4 documentation_. PyTorch Lightning Documentation — PyTorch Lightning 1.1.4 documentation. [https://pytorch-lightning.readthedocs.io/en/stable/lightning\_module.html](https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html)

PyTorch Lightning. (n.d.). _Trainer — PyTorch lightning 1.1.4 documentation_. PyTorch Lightning Documentation — PyTorch Lightning 1.1.4 documentation. [https://pytorch-lightning.readthedocs.io/en/latest/trainer.html](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html)

PyTorch. (n.d.). _CUDA semantics — PyTorch 1.7.0 documentation_. [https://pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)

PyTorch Lightning. (n.d.). _Multi-GPU training — PyTorch lightning 1.1.4 documentation_. PyTorch Lightning Documentation — PyTorch Lightning 1.1.4 documentation. [https://pytorch-lightning.readthedocs.io/en/latest/multi\_gpu.html](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html)
