---
title: "How to generate a summary of your Keras model?"
date: "2020-04-01"
categories:
  - "deep-learning"
  - "frameworks"
tags:
  - "deep-learning"
  - "keras"
  - "model-summary"
  - "neural-network"
  - "summary"
---

When building a neural network with the Keras framework for deep learning, I often want to have a quick and dirty way of checking whether everything is all right. That is, whether my layers output data correctly, whether my parameters are in check, and whether I have a good feeling about the model as a whole.

Keras model summaries help me do this. They provide a text-based overview of what I've built, which is especially useful when I have to add symmetry such as with autoencoders. But how to create these summaries? And why are they so useful? We'll discover this in today's blog post.

Firstly, we'll look at some high-level building blocks which I usually come across when I build neural networks. Then, we continue by looking at how Keras model summaries help me during neural network development. Subsequently, we generate one ourselves, by adding it to an example Keras ConvNet. This way, you'll be able to generate model summaries too in your Keras models.

Are you ready? Let's go! 😊

* * *

\[toc\]

* * *

## High-level building blocks of a Keras model

I've created quite a few neural networks over the past few years. While everyone has their own style in creating them, I always see a few high-level building blocks return in my code. Let's share them with you, as it will help you understand the model with which we'll be working today.

First of all, you'll always state **the imports of your model**. For example, you import Keras - today often as `tensorflow.keras.something`, but you'll likely import Numpy, Matplotlib and other libraries as well.

Next, and this is entirely personal, you'll find the **model configuration**. The model compilation and model training stages - which we'll cover soon - require configuration. This configuration is then spread across a number of lines of code, which I find messy. That's why I always specify a few Python variables storing the model configuration, so that I can refer to those when I actually configure the model.

Example variables are the batch size, the size of your input data, your [loss function](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/), the [optimizer](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/) that you will use, and so on.

Once the model configuration was specified, you'll often **load and preprocess your dataset**. Loading the dataset can be done in a multitude of ways - you can load data from file, you can use the [Keras datasets](https://www.machinecurve.com/index.php/2019/12/31/exploring-the-keras-datasets/), it doesn't really matter. Below, we'll use the latter scenario. Preprocessing is done in a minimal way - in line with the common assumption within the field of deep learning that models will take care of feature extraction themselves as much as possible - and often directly benefits the training process.

Once data is ready, you next **specify the architecture of your neural network**. With Keras, you'll often use the Sequential API, because it's easy. It allows you to stack individual layers on top of each other simply by calling `model.add`.

Specifying the architecture actually means creating the skeleton of your neural network. It's a design, rather than an actual model. To make an actual model, we move to the **model compilation** step - using `model.compile`. Here, we actually _instantiate_ the model with all the settings that we configured before. Once compiled, we're ready to start training.

**Starting the training process** is what we finally do. By using `model.fit`, we fit the dataset that we're training with to the model. The training process should now begin as configured by yourself.

Finally, once training has finished, you wish to **evaluate** the model against data that it hasn't yet seen - to find out whether it _really_ performs and did not simply [overfit](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/) to your training set. We use `model.evaluate` for this purpose.

* * *

## Model summaries

...what is lacking, though, is some quick and dirty information about your model. Can't we generate some kind of summary?

Unsurprisingly, we can! 😀 It would look like this:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 64)        18496
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 26, 26, 128)       73856
_________________________________________________________________
flatten (Flatten)            (None, 86528)             0
_________________________________________________________________
dense (Dense)                (None, 128)               11075712
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 11,170,250
Trainable params: 11,170,250
Non-trainable params: 0
_________________________________________________________________
```

There are multiple benefits that can be achieved from generating a model summary:

- Firstly, you have that quick and dirty overview of the components of your Keras model. The names of your layers, their types, as well as the shape of the data that they output and the number of trainable parameters.
- Secondly, with respect to the shape of your output data, this is beneficial if - for example - you have a mismatch somewhere. This can happen in the case of an [autoencoder](https://www.machinecurve.com/index.php/2019/12/19/creating-a-signal-noise-removal-autoencoder-with-keras/), where you effectively link two funnels together in order to downsample and upsample your data. As you want to have perfect symmetry, model summaries can help here.
- Thirdly, with respect to the number of parameters, you can make a guess as to where overfitting is likely and why/where you might face computational bottlenecks. The more trainable parameters your model has, the more computing power you need. What's more, if you provide an overkill of trainable parameters, your model might also be more vulnerable to overfitting, especially when the total size of your model or the size of your dataset does not account for this.

Convinced? Great 😊

* * *

## Generating a model summary of your Keras model

Now that we know some of the high-level building blocks of a Keras model, and know how summaries can be beneficial to understand your model, let's see if we can actually generate a summary!

For this reason, we'll give you an example [Convolutional Neural Network](https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/) for two-dimensional inputs. Here it is:

```python
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 25
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# Load CIFAR-10 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Scale data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(input_train, target_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

Clearly, all the high-level building blocks are visible:

- The imports speak for themselves.
- The model configuration variables tell us that we'll be using [sparse categorical crossentropy loss](https://www.machinecurve.com/index.php/2019/10/06/how-to-use-sparse-categorical-crossentropy-in-keras/) and the [Adam optimizer](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/). We will train for ten epochs (or iterations) and feed the model 50 samples at once.
- We load the CIFAR10 dataset, which contains everyday objects - see below for some examples. Once it's loaded, we do two three things: firstly, we'll determine the shape of our data - to be used in the first model layer. Secondly, we cast the numbers into `float32` format, which might speed up the training process when you are using a GPU powered version of Keras. Thirdly, and finally, we scale the data, to ensure that we don't face massive weight swings during the optimization step after each iteration. As you can see, we don't really do feature engineering _in terms of the features themselves_, but rather, we do some things to benefit the training process.
- We next specify the model architecture: three [Conv2D layers](https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/) for feature extraction, followed by a Flatten layer, as our Dense layers - which serve to generate the classification - can only handle one-dimensional data.
- Next, we compile the skeleton into an actual model and subsequently start the training process.
- Once training has finished, we evaluate and show the evaluation on screen.

[![](images/cifar10_images.png)](https://www.machinecurve.com/wp-content/uploads/2019/11/cifar10_images.png)

Now, how to add that summary?

Very simple.

Add `model.summary()` to your code, perhaps with a nice remark, like `# Display a model summary`. Like this:

```python
# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Display a model summary
model.summary()

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])
```

Running the model again then nicely presents you the model summary:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 64)        18496
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 26, 26, 128)       73856
_________________________________________________________________
flatten (Flatten)            (None, 86528)             0
_________________________________________________________________
dense (Dense)                (None, 128)               11075712
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 11,170,250
Trainable params: 11,170,250
Non-trainable params: 0
_________________________________________________________________
```

Nice! 🎆

* * *

## Summary

In this blog post, we looked at generating a model summary for your Keras model. This summary, which is a quick and dirty overview of the layers of your model, display their output shape and number of trainable parameters. Summaries help you debug your model and allow you to immediately share the structure of your model, without having to send all of your code.

For this to work, we also looked at some high-level components of a Keras based neural network that I often come across when building models. Additionally, we provided an example ConvNet to which we added a model summary.

Although it's been a relatively short blog post, I hope that you've learnt something today! If you did, or didn't, or when you have questions/remarks, please leave a comment in the comments section below. I'll happily answer your comment and improve my blog post where necessary.

Thank you for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

* * *

## References

_Keras_. (n.d.). Utils: Model Summary. [https://keras.io/utils/#print\_summary](https://keras.io/utils/#print_summary)
