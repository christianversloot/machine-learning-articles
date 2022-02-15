---
title: "Using simple generators to flow data from file with Keras"
date: "2020-04-06"
categories: 
  - "deep-learning"
  - "frameworks"
tags: 
  - "big-data"
  - "dataset"
  - "deep-learning"
  - "generator"
  - "keras"
  - "large-dataset"
  - "machine-learning"
---

During development of basic neural networks - such as the ones we build to show you how e.g. [Conv2D layers work](https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/) - we often load the whole dataset into memory. This is perfectly possible, because the datasets we're using are relatively small. For example, the MNIST dataset has only 60.000 samples in its _training_ part.

Now, what if datasets are larger? Say, they are 1.000.000 samples, or even more? At some point, it might not be feasible or efficient to store all your samples in memory. Rather, you wish to 'stream' them from e.g. a file. How can we do this with Keras models? That's what we will cover in today's blog post.

Firstly, we'll take a look at the question as to why: why flow data from a file anyway? Secondly, we'll take a look at _generators_ - and more specifically, _custom_ generators. Those will help you do precisely this. In our discussion, we'll also take a look at how you must fit generators to TensorFlow 2.x / 2.0+ based Keras models. Finally, we'll give you an example - how to fit data from a _very simple CSV file_ with a generator instead of directly from memory.

**Update 05/Oct/2020:** provided example of using generator for validation data with `model.fit`.

* * *

\[toc\]

* * *

## Why would you flow data from a file?

The answer is really simple: sometimes, you don't want to spend all your memory storing the data.

You wish to use some of that memory for other purposes, too.

In that case, fitting the data with a custom generator can be useful.

### Fitting data with a custom generator

But what is such a generator? For this, we'll have to look at the Python docs:

> Generator functions allow you to declare a function that behaves like an iterator, i.e. it can be used in a for loop.
> 
> [Python (n.d.)](https://wiki.python.org/moin/Generators)

It already brings us further, but it's still vague, isn't it?

Ilya Michlin (2019) explains the need for generators in better terms, directly related to machine learning:

> You probably encountered a situation where you try to load a dataset but there is not enough memory in your machine. As the field of machine learning progresses, this problem becomes more and more common. Today this is already one of the challenges in the field of vision where large datasets of images and video files are processed.
> 
> [Michlin (2019)](https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c)

Combined with the rather vague explanation, we can get there.

A generator can be used to "behave like an iterator", "used in a loop" - to get us small parts of some very large data file. These parts, in return, can subsequently be fed to the model for training, to avoid the memory problems that are common in today's machine learning projects.

Bingo! Generators can help us train with large data files. Nice :)

* * *

## Example model

Let's now take a look at an example with Keras. Suppose that we have this massive but simple dataset - 500.000.000 rows of simple \[latex\]x \\rightarrow y\[/latex\] mappings:

```

x,y
1,1
2,2
3,3
4,4
5,5
...
```

This file might be called e.g. `five_hundred.csv`.

As you might expect, this is the linear function \[latex\]y: f(x) = x\[/latex\]. It's one of the most simple regression scenarios that you can encounter.

Now, let's build a model for this dataset just like we always do - with one exception: we use a generator to load the data rather than loading it in memory. Here's what we'll do:

- We load our imports, which represent the dependencies for today's model;
- We set some basic configuration options - specifically targeted at the dataset that we'll be feeding today;
- We specify the function which loads the data;
- We create the - very simple - model architecture;
- We compile the model;
- We fit the generator to the model.

Let's go!

### Loading our imports

As always, the first thing we do is loading our imports. We import the Sequential API from `tensorflow.keras`, the TensorFlow 2.x way of importing Keras, as well as the Dense layer. As you may understand by now, we'll be building a densely-connected neural network with the Sequential API. Additionally, we also import TensorFlow itself, and Numpy.

```
# Load dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow
```

### Setting some basic configuration options

Next, we set two configuration options. First, we specify the total number of rows present in the file:

```
# Num rows
num_rows = 5e8 # Five hundred million
batch_size = 250
```

`5e8` equals `500e6` which equals 500.000.000.

Additionally, we feed 250 samples in a minibatch during each iteration. By consequence, in this case, we'll have `2e6` or 2.000.000 steps of 250 samples per epoch, as we will see later.

### Specifying the function that loads the data

Now that we have configured our model, we can specify the function that loads the data. It's actually a pretty simple function - a simple Python definition.

It has a `path` and a `batchsize` attribute, which are used later, and first creates empty arrays with `inputs` and `targets`. What's more, it sets the `batchcount` to 0. Why this latter one is necessary is what we will see soon.

Subsequently, we keep iterating - we simply set `while True`, which sets a never-ending loop until the script is killed. Every time, we we open the file, and subsequently parse the inputs and targets. Once the batch count equals the batch size that we configured (do note that this happens when we have the _exact same size_, as the batch count starts at 0 instead of 1), we finalize the arrays and subsequently yield the data. Don't forget to reset the `inputs`, `targets` and `batchcount`, though!

```
# Load data
def generate_arrays_from_file(path, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    while True:
        with open(path) as f:
            for line in f:
                x,y = line.split(',')
                inputs.append(x)
                targets.append(y)
                batchcount += 1
                if batchcount > batchsize:
                  X = np.array(inputs, dtype='float32')
                  y = np.array(targets, dtype='float32')
                  yield (X, y)
                  inputs = []
                  targets = []
                  batchcount = 0
```

### Creating the model architecture

Now that we have specified our function for flowing data from file, we can create the architecture of our model. Today, our architecture will be pretty simple. In fact, it'll be a three-layered model, of which two layers are hidden - the latter one is the output layer, and the input layer is specified implicitly.

As you can see by the number of output neurons for every layer, slowly but surely, an information bottleneck is created. We use [ReLU](https://www.machinecurve.com/index.php/2019/09/04/relu-sigmoid-and-tanh-todays-most-used-activation-functions/) for activating in the hidden layers, and `linear` for the final layer. This, in return, suggests that we're dealing with a regression scenario. Unsurprisingly: we are.

```
# Create the model
model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
```

### Compiling the model

This latter fact gets even more clear when we look at the `compile` function for our model. As our loss, we use the mean absolute error, which is a typical [loss function for regression problems](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#loss-functions-for-regression). Additionally, we specify the mean squared error, which is one too. Adam is [used for optimizing the model](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/#adam) - which is a common choice, especially when you don't really care about optimizers, as we do now (it's not the goal of today's blog post), Adam is an adequate choice.

```
# Compile the model
model.compile(loss='mean_absolute_error',
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['mean_squared_error'])
```

### Fitting the generator

Next, we `fit` the generator function - together with the file and batch size - to the model. This will allow the data to flow from file into the model directly.

Note that on my machine, this file with five hundred million rows exceeds 10GB. If it were bigger, it wouldn't have fit in memory!

```
# Fit data to model
model.fit(generate_arrays_from_file('./five_hundred.csv', batch_size),
                    steps_per_epoch=num_rows / batch_size, epochs=10)
```

### Full model code

Altogether, here's the code as a whole:

```
# Load dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow

# Num rows
num_rows = 5e8 # Five hundred million
batch_size = 250

# Load data
def generate_arrays_from_file(path, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    while True:
        with open(path) as f:
            for line in f:
                x,y = line.split(',')
                inputs.append(x)
                targets.append(y)
                batchcount += 1
                if batchcount > batchsize:
                  X = np.array(inputs, dtype='float32')
                  y = np.array(targets, dtype='float32')
                  yield (X, y)
                  inputs = []
                  targets = []
                  batchcount = 0

# Create the model
model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_absolute_error',
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['mean_squared_error'])

# Fit data to model
model.fit(generate_arrays_from_file('./five_hundred.csv', batch_size),
                    steps_per_epoch=num_rows / batch_size, epochs=10)
```

Running it does indeed start the training process, but it will take a while:

```
Train for 2000000.0 steps
Epoch 1/10
2020-04-06 20:33:35.556364: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
 352683/2000000 [====>.........................]
```

However, we successfully completed our code! 🎉

* * *

## Using model.fit using validation data specified as a generator

In the model above, I'm working with a generator that flows training data into the model during the forward pass executed by the `.fit` operation when training. Unsurprisingly, some people have asked in the comments section of this post if it is possible to use a generator for validation data too, and if so, how.

Remember that validation data is used during the training process in order to identify whether the machine learning model [has started overfitting](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/#how-well-does-your-model-perform-underfitting-and-overfitting). Testing data, in the end, is used to test whether your model generalizes to data that it hasn't seen before.

I thought this wouldn't be possible, as the TensorFlow documentation clearly states this:

>  Note that `validation_data` does not support all the data types that are supported in `x`, eg, dict, generator or [`keras.utils.Sequence`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence).
> 
> _Tf.keras.Model_. (n.d.)

Here, `x` is the input argument, or the training data you're inputting - for which we used a generator above.

One comment to this post argued that counter to the docs, it is possible to use a generator for validation data. And indeed, it seems to work (Stack Overflow, n.d.). That's why it's also possible to let validation data flow into your model - like this:

```
# Fit data to model
model.fit(generate_arrays_from_file('./five_hundred.csv', batch_size),
                    steps_per_epoch=num_rows / batch_size, epochs=10, validation_data=generate_arrays_from_file('./five_hundred_validation_split.csv', batch_size), validation_steps=num_val_steps)
```

Here:

- We use the same generator for our validation set, which however comes from a different file (`./five_hundred_validation_split.csv`). Do note that we're using the same batch size, but this can be a different one as validation happens once training has finished.
- Do note that normally, TensorFlow infers the number of `validation_steps` (i.e. how many batches of samples are used for validation) automatically by means of the rule `if None then len(validation_set)/batch_size`. However, the length of our validation set is not known up front, because it's generated from file. We must specify it manually. We should therefore add a number of validation steps manually; the value is dependent on the length of your validation set. If it has one million rows, it's best to set it to one million, e.g. `num_val_steps = int(1e6/batch_size)`. If your `batch_size` is 250, the number of validation steps would be 4.000.

Thanks to PhilipT in the comments section for reporting! :)

* * *

## Summary

In this blog post, we looked at the concept of generators - and how we can use them with Keras to overcome the problem of data set size. More specifically, it allows us to use really large datasets when training our Keras model. In an example model, we showed you how to use generators with your Keras model - in our example, with a data file of more than 10GB.

I hope you've learnt something today! If you did, please feel free to leave a comment in the comments section below. Make sure to feel welcome to do the same if you have questions, remarks or suggestions for improvement. Where possible, I will answer and adapt the blog post! :)

Thank you for reading MachineCurve today and happy engineering 😎

\[kerasbox\]

* * *

## References

Python. (n.d.). _Generators - Python Wiki_. Python Software Foundation Wiki Server. Retrieved April 6, 2020, from [https://wiki.python.org/moin/Generators](https://wiki.python.org/moin/Generators)

Michlin, I. (2019, October 6). _Keras data generators and how to use them_. Medium. [https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c](https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c)

Keras. (n.d.). _Sequential_. Home - Keras Documentation. [https://keras.io/models/sequential/#fit\_generator](https://keras.io/models/sequential/#fit_generator)

_Could validation data be a generator in tensorflow.keras 2.0?_ (n.d.). Stack Overflow. [https://stackoverflow.com/questions/60003006/could-validation-data-be-a-generator-in-tensorflow-keras-2-0](https://stackoverflow.com/questions/60003006/could-validation-data-be-a-generator-in-tensorflow-keras-2-0)

_Tf.keras.Model_. (n.d.). TensorFlow. [https://www.tensorflow.org/api\_docs/python/tf/keras/Model#fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
