---
title: "How to visualize a model with TensorFlow 2 and Keras?"
date: "2019-10-07"
categories: 
  - "buffer"
  - "frameworks"
tags: 
  - "architecture"
  - "deep-learning"
  - "keras"
  - "neural-network"
  - "visualization"
---

Every now and then, you might need to demonstrate your Keras model structure. There's one or two things that you may do when this need arises. First, you may send the person who needs this overview your code, requiring them to derive the model architecture themselves. If you're nicer, you send them a model of your architecture.

...but creating such models is often a hassle when you have to do it manually. Solutions like www.draw.io are used quite often in those cases, because they are (relatively) quick and dirty, allowing you to create models fast.

However, there's a better solution: the built-in `plot_model` facility within Keras. It allows you to create a visualization of your model architecture. In this blog, I'll show you how to create such a visualization. Specifically, I focus on the model itself, discussing its architecture so that you fully understand what happens. Subsquently, I'll list some software dependencies that you'll need - including a highlight about a bug in Keras that results in a weird error related to `pydot` and GraphViz, which are used for visualization. Finally, I present you the code used for visualization and the end result.

**After reading this tutorial, you will...**

- Understand what the `plot_model()` util in TensorFlow 2.0/Keras does.
- Know what types of plots it generates.
- Have created a neural network that visualizes its structure.

_Note that model code is also available [on GitHub](https://github.com/christianversloot/keras-visualizations)._

* * *

**Update 22/Jan/2021:** ensured that the tutorial is up-to-date and reflects code for TensorFlow 2.0. It can now be used with recent versions of the library. Also performed some header changes and textual improvements based on the switch from Keras 1.0 to TensorFlow 2.0. Also added an exampl of horizontal plotting.

* * *

\[toc\]

* * *

## Code example: using plot\_model for visualizing the model

If you want to get started straight away, here is the code that you can use for visualizing your TensorFlow 2.0/Keras model with `plot_model`:

```
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')
```

Make sure to read the rest of this tutorial if you want to understand everything in more detail!

* * *

## Today's to-be-visualized model

To show you how to visualize a Keras model, I think it's best if we discussed one first.

Today, we will visualize the [Convolutional Neural Network](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/) that we created earlier to demonstrate the benefits of using CNNs over densely-connected ones.

This is the code of that model:

```
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Set input shape
sample_shape = input_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# Reshape data 
input_train = input_train.reshape(len(input_train), input_shape[0], input_shape[1], input_shape[2])
input_test  = input_test.reshape(len(input_test), input_shape[0], input_shape[1], input_shape[2])

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Convert them into black or white: [0, 1].
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = tensorflow.keras.utils.to_categorical(target_train, no_classes)
target_test = tensorflow.keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

What does it do?

I'd suggest that you [read the post](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/) if you wish to understand it very deeply, but I'll briefly cover it here.

It simply classifies the MNIST dataset. This dataset contains 28 x 28 pixel images of digits, or numbers between 0 and 9, and our CNN classifies them with a staggering 99% accuracy. It does so by combining two convolutional blocks (which consist of a two-dimensional convolutional layer, two-dimensional max pooling and dropout) with densely-conneted layers. It's the best of both worlds in terms of interpreting the image _and_ generating final predictions.

But how to visualize this model's architecture? Let's find out.

* * *

## Built-in `plot_model` util

Utilities. I love them, because they make my life easier. They're often relatively simple functions that can be called upon to perform some relatively simple actions. Don't be fooled, however, because these actions often benefit one's efficiently greatly - in this case, not having to visualize a model architecture yourself in tools like draw.io

I'm talking about the `plot_model` util, which comes [delivered with Keras](https://keras.io/visualization/#model-visualization).

It allows you to create a visualization of your Keras neural network.

More specifically, the Keras docs define it as follows:

```
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')
```

From the **Keras utilities**, one needs to import the function, after which it can be used with very minimal parameters:

- The **model instance**, or the model that you created - whether you created it now or preloaded it instead from a model [saved to disk](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/).
- And the `to_file` parameter, which essentially specifies a location on disk where the model visualization is stored.

If you wish, you can supply some additional parameters as well:

- The **show\_shapes** argument (which is `False` by default) which controls whether the _shape_ of the layer outputs are shown in the graph. This would be beneficial if besides the architecture you also need to understand _how it transforms data_.
- With **show\_dtypes** (`False` by default) you can indicate whether to show layer data types on the plot.
- The **show\_layer\_names** argument (`True` by default) which determines whether the names of the layers are displayed.
- The **rankdir** (`TB` by default) can be used to indicate whether you want a vertical or horizontal plot. `TB` is vertical, `LR` is horizontal.
- The **expand\_nested** (`False` by default) controls how nested models are displayed.
- **Dpi** controls the dpi value of the image.

However, likely, for a simple visualization, you don't need them. Let's now take a look what we would need if we were to create such a visualization.

* * *

## Software dependencies

If you wish to run the code presented in this blog successfully, you need to install certain software dependencies. You'll need those to run it:

- **TensorFlow 2.0**, or any subsequent version, which makes sense given the fact that we're using a Keras util for model visualization;
- **Python**, preferably 3.8+, which is required if you wish to run Keras.
- **Graphviz**, which is a graph visualization library for Python. Keras uses it to generate the visualization of your neural network. [You can install Graphviz from their website.](https://graphviz.gitlab.io/download/)

Preferably, you'll run this from an **Anaconda** environment, which allows you to run these packages in an isolated fashion. Note that many people report that a `pip` based installation of Graphviz doesn't work; rather, you'll have to install it separately into your host OS from their website. Bummer!

### Keras bug: `pydot` failed to call GraphViz

When trying to visualize my Keras neural network with `plot_model`, I ran into this error:

```
'`pydot` failed to call GraphViz.'
OSError: `pydot` failed to call GraphViz.Please install GraphViz (https://www.graphviz.org/) and ensure that its executables are in the $PATH.
```

...which essentially made sense at first, because I didn't have Graphviz installed.

...but which didn't after I installed it, because the error kept reappearing, even after restarting the Anaconda terminal.

Sometimes, it helps to install `pydotplus` as well with `pip install pydotplus`. Another solution, although not preferred, is to downgrade your `pydot` version.

* * *

## Visualization code

When adapting the code from my original CNN, scrapping away the elements I don't need for visualizing the model architecture, I end up with this:

```
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Set input shape
sample_shape = input_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# Number of classes
no_classes = 10

# Reshape data 
input_train = input_train.reshape(len(input_train), input_shape[0], input_shape[1], input_shape[2])
input_test  = input_test.reshape(len(input_test), input_shape[0], input_shape[1], input_shape[2])

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

plot_model(model, to_file='model.png')
```

You'll first perform the **imports** that you still need in order to successfully run the Python code. Specifically, you'll import the Keras library, the Sequential API and certain layers - this is obviously dependent on what _you_ want. Do you want to use the Functional API? That's perfectly fine. Other layers? Fine too. I just used them since the CNN is exemplary.

Note that I also imported `plot_model` with `from tensorflow.keras.utils import plot_model` and reshaped the data to accomodate for the Conv2D layer.

Speaking about architecture: that's what I kept in. Based on the Keras Sequential API, I apply the two convolutional blocks as discussed previously, before flattening their output and feeding it to the densely-connected layers generating the final prediction. And, of course, we need `no_classes = 10` to ensure that our final `Dense` layer works as well.

However, in this case, no such prediction is generated. Rather, the `model` instance is used by `plot_model` to generate a model visualization stored at disk as `model.png`. Likely, you'll add hyperparameter tuning and data fitting later on - but hey, that's not the purpose of this blog.

* * *

## End result

And your final end result looks like this:

![](images/model.png)

* * *

## Making a horizontal TF/Keras model plot

Indeed, above we saw that we can use the `rankdir` attribute (which is set to `TB` i.e. vertical by default) to generate a horizontal plot! This is new, and highly preferred, as we sometimes don't want these massive vertical plots.

Making a horizontal plot of your TensorFlow/Keras model simply involves adding the `rankdir='LR'` a.k.a. _horizontal_ attribute:

```
plot_model(model, to_file='model.png', rankdir='LR')
```

Which gets you this:

[![](images/model.png)](https://www.machinecurve.com/wp-content/uploads/2021/01/model.png)

Awesome!

* * *

## Summary

In this blog, you've seen how to create a Keras model visualization based on the `plot_model` util provided by the library. I hope you found it useful - let me know in the comments section, I'd appreciate it! 😎 If not, let me know as well, so I can improve. For now: happy engineering! 👩‍💻

_Note that model code is also available [on GitHub](https://github.com/christianversloot/keras-visualizations)._

* * *

## References

How to create a CNN classifier with Keras? – MachineCurve. (2019, September 24). Retrieved from [https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/)

Keras. (n.d.). Visualization. Retrieved from [https://keras.io/visualization/](https://keras.io/visualization/)

Avoid wasting resources with EarlyStopping and ModelCheckpoint in Keras – MachineCurve. (2019, June 3). Retrieved from [https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/)

pydot issue · Issue #7 · XifengGuo/CapsNet-Keras. (n.d.). Retrieved from [https://github.com/XifengGuo/CapsNet-Keras/issues/7#issuecomment-536100376](https://github.com/XifengGuo/CapsNet-Keras/issues/7#issuecomment-536100376)

TensorFlow. (2021). _Tf.keras.utils.plot\_model_. [https://www.tensorflow.org/api\_docs/python/tf/keras/utils/plot\_model](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)
