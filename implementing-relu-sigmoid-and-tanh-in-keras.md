---
title: "ReLU, Sigmoid and Tanh with TensorFlow 2 and Keras"
date: "2019-09-09"
categories: 
  - "buffer"
  - "deep-learning"
  - "frameworks"
tags: 
  - "activation-functions"
  - "deep-learning"
  - "keras"
  - "relu"
  - "sigmoid"
  - "tanh"
---

In a recent tutorial, we looked at [widely used activation functions](https://machinecurve.com/index.php/2019/09/04/relu-sigmoid-and-tanh-todays-most-used-activation-functions/) in today's neural networks. More specifically, we checked out Rectified Linear Unit (ReLU), Sigmoid and Tanh (or hyperbolic tangent), together with their benefits and drawbacks.

However, it all remained theory.

In this blog post, we'll move towards implementation. Because how to build up neural networks with ReLU, Sigmoid and tanh in Keras, one of today's popular deep learning frameworks?

If you're interested in the inner workings of the activation functions, check out the link above.

If you wish to implement them, make sure to read on! 😎

In this tutorial, you will...

- Understand what the ReLU, Tanh and Sigmoid activations are.
- See where to apply these activation functions in your TensorFlow 2.0 and Keras model.
- Walk through an end-to-end example of implementing ReLU, Tanh or Sigmoid in your Keras model.

Note that the results are [also available on GitHub](https://github.com/christianversloot/relu-tanh-sigmoid-keras).

* * *

**Update 18/Jan/2021:** ensure that the tutorial is up to date for 2021. Also revisited header information.

**Update 03/Nov/2020:** made code compatible with TensorFlow 2.x.

* * *

\[toc\]

* * *

## Code examples: using ReLU, Tanh and Sigmoid with TF 2.0 and Keras

These code examples show how you can add ReLU, Sigmoid and Tanh to your TensorFlow 2.0/Keras model. If you want to understand the activation functions in more detail, or see how they fit in a Keras model as a whole, make sure to continue reading!

### Rectified Linear Unit (ReLU)

```
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
```

### Sigmoid

```
model.add(Dense(12, input_shape=(8,), activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
```

### Tanh

```
model.add(Dense(12, input_shape=(8,), activation='tanh'))
model.add(Dense(8, activation='tanh'))
```

* * *

## Recap: ReLU, Tanh and Sigmoid

Before we begin, a small recap on the concept of an activation function and the [three widely ones used today](https://machinecurve.com/index.php/2019/09/04/relu-sigmoid-and-tanh-todays-most-used-activation-functions/).

Neural networks are composed of layers of individual neurons which can take vector data as input and subsequently either fire to some extent or remain silent.

Each individual neuron multiplies an input vector with its weights vector to compute the so-called dot product, subsequently adding a bias value, before emitting the output.

However, the multiplication and addition operations are linear and by consequence when applied neural networks can only handle linear data well.

This is not desirable because most real-world data is nonlinear in nature. For example, it's really hard to draw a line through an image to separate an object from its surroundings.

Hence, activation functions are applied to neural networks: the linear output is first input into such a function before being emitted to the next layer. Since activation functions are nonlinear, the linear input will be transformed into nonlinear output. When applied to all neurons, the system as a whole becomes nonlinear, capable of learning from highly complex, nonlinear data.

ReLU, Sigmoid and Tanh are today's most widely used activation functions. From these, ReLU is the most prominent one and the de facto standard one during deep learning projects because it is resistent against the [vanishing and exploding gradients](https://machinecurve.com/index.php/2019/08/30/random-initialization-vanishing-and-exploding-gradients/) problems, whereas Sigmoid and Tanh are not. Hence, it's good practice to start with ReLU and expand from there. However, this must always be done with its challenges in mind: ReLU is not perfect and is [continuously improved](https://machinecurve.com/index.php/2019/05/30/why-swish-could-perform-better-than-relu/).

Now that we have a little background on these activation functions, we can introduce the dataset we're going to use to implement neural networks with ReLU, Sigmoid and Tanh in Keras.

* * *

## Today's dataset

Today, we're going to use a dataset that we used before when discussing [Rosenblatt Perceptrons and Keras](https://machinecurve.com/index.php/2019/07/24/why-you-cant-truly-create-rosenblatts-perceptron-with-keras/): the **Pima Indians Diabetes Database**.

This is what it does:

> This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
> 
> Source: [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

The nice thing about this dataset is that it is relatively simple. Hence, we can fully focus on the implementation rather than having to be concerned about data related issues. Additionally, it is freely available at [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database), under a CC0 license. This makes it the perfect choice for a blog like this.

The dataset very simply tries to predict the following:

- **Outcome:** Whether a person has diabetes (1) or not (0), the 0 and 1 being the target values.

For machine learning projects, it allows you to find correlations between (combinations of) those input values and the target values:

- **Pregnancies:** the number of times one has been pregnant;
- **Glucose:** one's plasma glucose concentration;
- **BloodPressure:** one's diastolic (lower) blood pressure value in mmHg.
- **SkinThickness:** the thickness of one's skin fold at the triceps, in mm.
- **Insulin:** one's 2-hour serum insulin level;
- **BMI:** one's Body Mass Index;
- **Diabetes pedigree function:** one's sensitivity to diabetes e.g. based on genetics;
- **Age:** one's age in years.

* * *

## General model parts

Today, we'll build a very simple model to illustrate our point. More specifically, we will create a [multilayer perceptron](https://machinecurve.com/index.php/2019/07/30/creating-an-mlp-for-regression-with-keras/) with Keras - but then three times, each time with a different activation function.

To do this, we'll start by creating three files - one per activation function: `relu.py`, `sigmoid.py` and `tanh.py`. In each, we'll add general parts that are shared across the model instances.

Note that you'll need the dataset as well. You could either download it from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) or take a look at GitHub, where it is present as well. Save the `pima_dataset.csv` file in the same folder as your `*.py` files.

We begin with the dependencies:

```
# Load dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
```

They are really simple today. We use the Keras Sequential API, which is the simplest of two and allows us to add layers sequentially, or in a line. We also import the `Dense` layer, which is short for densely-connected, or the layer types that are traditionally present in a multilayer perceptron.

Additionally, we import `numpy` for reading the file and preparing the dataset.

Second, we load the data:

```
# Load data
dataset = np.loadtxt('./pima_dataset.csv', delimiter=',')
```

Since the data is comma-separated, we set the `delimiter` to a comma.

We then separate the input data and the target data:

```
# Separate input data and target data
X = dataset[:, 0:8]
Y = dataset[:, 8]
```

In the CSV file, the data is appended together. That is, each row contains both the input data (the data used for training) and the outcomes (0/1) that are related to the input data. They need to be split if we want to train the model. We do so with the code above. It essentially takes 8 columns and makes it input data (columns 0-7), and one (the 8th) as target data.

We then start off with the model itself and instantiate the Sequential API:

```
# Create the Perceptron
model = Sequential()
```

We're then ready to add some activation function-specific code. We'll temporarily indicate its position with a comment:

```
# ActivationFunction-specific code here
```

...and continue with our final general steps:

```
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model 
model.fit(X, Y, epochs=225, batch_size=25, verbose=1, validation_split=0.2)
```

What we do first is adding the _final_ layer in the model: a Dense layer with one neuron and a Sigmoid activation function. This is what we need: since our classification problem is binary, we need one output neuron (that outputs a value between class 0 and class 1). The [Sigmoid](https://machinecurve.com/index.php/2019/09/04/relu-sigmoid-and-tanh-todays-most-used-activation-functions/) activation function allows us to do exactly that. Hence, we use it in our final layer too.

Compiling the model with binary crossentropy (we have a binary classification problem), the Adam optimizer (an extension of stochastic gradient descent that allows local parameter optimization and adds momentum) and accuracy is what we do second.

We finally fit the data (variables `X` and `Y` to the model), using 225 epochs with a batch size of 25. We set verbosity mode to 1 to see what happens and allow for a validation split of `0.2`: 20% of the data will be used for validating the training process after each epoch.

* * *

## Activation function-specific implementations

Now, it's time to add activation function-specific code. In all of the below cases, this is the part that you'll need to replace:

```
# ActivationFunction-specific code here
```

* * *

## TensorFlow 2.0 and Keras = Neural networks, made easy

As we may recall from the introduction of this blog _or_ the Keras website, this is the framework's goal:

> It was developed with a focus on enabling fast experimentation. _Being able to go from idea to result with the least possible delay is key to doing good research._

It is therefore no surprise that changing the activation function is very easy if you're using the standard ones. Essentially, Keras allows you to specify an activation function per layer by means of the `activation` parameter. As you can see above, we used this parameter to specify the Sigmoid activation in our final layer. The standard ones are available.

Today, Keras is tightly coupled to TensorFlow 2.0, and is still one of the key libraries for creating your neural networks. This article was adapted to reflect the latest changes in TensorFlow and works with any TensorFlow 2 version.

What's best, if the activation function of your choice - for example [Swish](https://machinecurve.com/index.php/2019/05/30/why-swish-could-perform-better-than-relu/) - is not available, you can create it yourself and add it as a function. Take a look at the Swish post to find an example.

### Adding ReLU to your model

By consequence, if we wish to implement a neural net work with ReLU, we do this:

```
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
```

### Adding Sigmoid to your model

...and with Sigmoid:

```
model.add(Dense(12, input_shape=(8,), activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
```

### Adding Tanh to your model

...or Tanh:

```
model.add(Dense(12, input_shape=(8,), activation='tanh'))
model.add(Dense(8, activation='tanh'))
```

Eventually, your code will look like this:

```
# Load dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load data
dataset = np.loadtxt('./pima_dataset.csv', delimiter=',')

# Separate input data and target data
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Create the Perceptron
model = Sequential()

# Model layers
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(X, Y, epochs=225, batch_size=25, verbose=1, validation_split=0.2)
```

The models are also available on [GitHub](https://github.com/christianversloot/relu-tanh-sigmoid-keras).

* * *

## Model performance: some observations

Let's now see if we can train the model. We can simply call the model by e.g. writing `python relu.py`, `python sigmoid.py` or `python tanh.py` depending on the model you wish to train.

Note that you need a fully operational deep learning environment to make it work. This means that you'll need Python (preferably 3.8+), that you'll need TensorFlow and Keras as well as Numpy. Preferably, you'll have this installed in an Anaconda container so that you have a pure deep learning environment for each time you're training.

When you have one, the training process starts upon execution of the command, and eventually this will be your output.

For ReLU:

```
Epoch 225/225
614/614 [==============================] - 0s 129us/step - loss: 0.4632 - acc: 0.7785 - val_loss: 0.5892 - val_acc: 0.7143
```

For Tanh:

```
Epoch 225/225
614/614 [==============================] - 0s 138us/step - loss: 0.5466 - acc: 0.7003 - val_loss: 0.6839 - val_acc: 0.6169
```

For Sigmoid:

```
Epoch 225/225
614/614 [==============================] - 0s 151us/step - loss: 0.5574 - acc: 0.7280 - val_loss: 0.6187 - val_acc: 0.7013
```

The results suggest that Tanh performs worse than ReLU and Sigmoid. This is explainable through the lens of its range: since we're having a binary classification problem, both Sigmoid and ReLU are naturally better suited for this task, particularly the Sigmoid function. Specifically, its binary crossentropy loss value is much higher than e.g. ReLU, although this one can also be improved much further - but that's not the point of this blog.

As you can see, it's always wise to consider multiple activation functions. In my master's thesis, I found that in some cases Tanh works better than ReLU. Since the practice of deep learning is often more art than science, it's always worth a try.

* * *

## Summary

In this blog, we've been introduced to activation functions and the most widely ones used today at a high level. Additionally, we checked the Pima Indians Diabetes Dataset and its contents and applied it with Keras to demonstrate how to create neural networks with the ReLU, Tanh and Sigmoid activation functions - [see GitHub](https://github.com/christianversloot/relu-tanh-sigmoid-keras). I hope you've found the answers to your challenges and hope you'll keep engineering! 😎

* * *

## References

TensorFlow. (2021). _Module: Tf.keras.activations_. [https://www.tensorflow.org/api\_docs/python/tf/keras/activations](https://www.tensorflow.org/api_docs/python/tf/keras/activations)

Keras. (n.d.). Activations. Retrieved from [https://keras.io/activations/](https://keras.io/activations/)

Kaggle. (n.d.). Pima Indians Diabetes Database. Retrieved from [https://www.kaggle.com/uciml/pima-indians-diabetes-database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
