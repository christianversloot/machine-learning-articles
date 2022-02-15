---
title: "How to use HDF5Matrix with Keras?"
date: "2020-04-26"
categories: 
  - "deep-learning"
  - "frameworks"
tags: 
  - "deep-learning"
  - "h5py"
  - "hdf5"
  - "hdf5matrix"
  - "keras"
  - "machine-learning"
  - "neural-networks"
  - "tensorflow"
---

In machine learning, when performing supervised learning, you'll have to load your dataset from somewhere - and then [feed it to the machine learning model](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process). Now, there are multiple ways for loading data.

A CSV file is one example, as well as a text file. It works really well if you're looking for simplicity: loading a dataset from a text-based file is really easy with Python.

The downside? Scalability. Reading text files is slow. And while this does not really matter when your dataset is small, it can become a true burden when you have dataset with millions and millions of rows.

As we've seen, the HDF5 format - the Hierarchical Data Format - comes to the rescue. This format, which stores the data into a hierarchy of groups and datasets (hence the name, plus version 5), is faster to read, as [we've seen before](https://www.machinecurve.com/index.php/2020/04/13/how-to-use-h5py-and-keras-to-train-with-data-from-hdf5-files/). It's also easily integrable with Python: with the `h5py` library, we can load our data into memory, and subsequently feed it to the machine learning training process.

Now, did you know that Keras already partially automates those steps? In fact, it does: the creators already provide a `util` that allows you to load a HDF5 based dataset easily, being the `HDF5Matrix`. Great!

In today's blog post, we'll take a look at this util. Firstly, we'll take a brief look at the HDF5 data format, followed by inspecting the HDF5Matrix util. Subsequently, we will provide an example of how to use it with an actual Keras model - both in preparing the dataset (should you need to, as we will see) and training the model.

Are you ready?

Let's go! :)

* * *

\[toc\]

* * *

## The HDF5 file format

Well, let's take a look at the HDF5 file format first - because we must know what we're using, before doing so, right?

Here we go:

> _Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data._
> 
> Wikipedia (2004)

As we can read in [our other blog post on HDF5](https://www.machinecurve.com/index.php/2020/04/13/how-to-use-h5py-and-keras-to-train-with-data-from-hdf5-files/), it is characterized as follows:

- It's a dataset that is designed for large datasets. Could be great for our ML projects!
- It consists of datasets and groups, where datasets are multidimensional arrays of a homogeneous type. Groups are container structures which can hold datasets and other groups.
- This way, we can group different sub datasets into one hierarchical structure, which we can transfer and interpret later. No more shuffling with CSV columns and delimiters, and so on. No - an efficient format indeed.

Here's a video for those who wish to understand HDF5 in more detail:

https://www.youtube.com/watch?v=q14F3WRwSck&feature=emb\_title

* * *

## The HDF5Matrix

Time to study the HDF5Matrix util. Well, it's not too exciting - haha :) In fact, this is what it is:

> Representation of HDF5 dataset which can be used instead of a Numpy array.
> 
> Keras (n.d.)

It's as simple as that.

In the Keras API, it is represented in two ways:

- In 'old' Keras i.e. `keras`, as `keras.utils.io_utils.HDF5Matrix`
- In 'new' Keras i.e. `tensorflow.keras`, as `tensorflow.keras.utils.HDF5Matrix`.

Take this into account when specifying your imports ;-)

Now, it also has a few options that can be configured by the machine learning engineer (Keras, n.d.):

- **datapath**: string, path to a HDF5 file
- **dataset**: string, name of the HDF5 dataset in the file specified in datapath
- **start**: int, start of desired slice of the specified dataset
- **end**: int, end of desired slice of the specified dataset
- **normalizer**: function to be called on data when retrieved

* * *

## Training a Keras model with HDF5Matrix

They pretty much speak for themselves, so let's move on to training a Keras model with HDF5Matrix.

### Today's dataset

Today's dataset will be the MNIST one, which we know pretty well by now - it's a numbers dataset of handwritten digits:

[![](images/mnist-visualize.png)](https://www.machinecurve.com/wp-content/uploads/2019/06/mnist-visualize.png)

Now, let’s take a look if we can create a simple [Convolutional Neural Network](https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/) which operates with the [MNIST dataset](https://www.machinecurve.com/index.php/2019/12/31/exploring-the-keras-datasets/#mnist-database-of-handwritten-digits), stored in HDF5 format.

Fortunately, this dataset is readily available at [Kaggle for download](https://www.kaggle.com/benedictwilkinsai/mnist-hd5f), so make sure to create an account there and download the **train.hdf5** and **test.hdf5** files.

### Why we need to adapt our data - and how to do it

Unfortunately - and this is the reason why we created that other blog post - the dataset cannot be used directly with a Keras model, for multiple reasons:

1. The shape of the input datasets (i.e. the training inputs and the testing inputs) is wrong. When image data is single-channel (and the MNIST dataset is), the dataset is often delivered without the channel dimension (in our case, that would equal a shape of `(60000, 28, 28)` for the training set instead of the desired `(60000, 28, 28, 1)`). If we were to use the data directly, we would get an error like `ValueError: Error when checking input: expected conv2d_input to have 4 dimensions, but got array with shape (60000, 28, 28)`.
2. The dataset is ill-prepared for neural networks, as it's unscaled. That is, the grayscale data has values somewhere between `[0, 255]` - that's the nature of grayscale data. Now, the distance between 0 and 255 is relatively far - especially so if we could also rescale the data so that the distance becomes much smaller, i.e. `[0, 1]`. While the relationships in the data won't change, the representation of the data does, and it really helps the training process - as it reduces the odds of weight swings during optimization.
3. What's more, it could also be cast into `float32` format, which presumably speeds up the training process on GPUs.

We will thus have to adapt our data. While the `HDF5Matrix` util provides the _normalizer_ function, it doesn't work when our data has the wrong shape - we still get that `ValueError`.

That's why we created that other blog post about applying `h5py` directly first.

#### Imports and configuration

But today, we'll make sure to adapt the data so that we can run it with `HDF5Matrix` too. Let's take a look. Make sure that `h5py` is installed with `pip install h5py`. Then open up a code editor, create a file such as `hdf5matrix_prepare.py` and write some code:

```
import h5py
```

This one speaks for itself. We import the `h5py` library.

```
# Configuration
img_width, img_height, img_num_channels = 28, 28, 1
```

This one does too. We set a few configuration options, being the image width, image height, and number of channels. As we know MNIST to be 28x28 px single-channel images, we set the values to 28, 28 and 1.

#### Loading the MNIST data

Then, we load the MNIST data:

```
# Load MNIST data
f = h5py.File('./train.hdf5', 'r')
input_train = f['image'][...]
label_train = f['label'][...]
f.close()
f = h5py.File('./test.hdf5', 'r')
input_test = f['image'][...]
label_test = f['label'][...]
f.close()
```

Here, we load the `image` and `label` datasets into memory for both the training and testing HDF5 files. The `[...]` part signals that we load the entire dataset into memory. If your dataset is too big, this might fail. You then might wish to rewrite the code so that you process the dataset in slices.

#### Reshaping, casting and scaling

Now that we have loaded the data, it's time to adapt our data to resolve the conflicts that we discussed earlier. First, we'll reshape the data:

```
# Reshape data
input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
input_test  = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))
```

The code speaks pretty much for itself. We set the shape to be equal to the size of the particular array, and the values for the image that we configured before.

Casting and scaling is also pretty straight-forward:

```
# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Scale data
input_train = input_train / 255
input_test = input_test / 255
```

#### Saving the adapted data

Then, we can save the data into new files - being `train_reshaped.hdf5` and `test_reshaped.hdf5`:

```
# Save reshaped training data
f = h5py.File('./train_reshaped.hdf5', 'w')
dataset_input = f.create_dataset('image', (len(input_train), img_width, img_height, img_num_channels))
dataset_label = f.create_dataset('label', (len(input_train),))
dataset_input[...] = input_train
dataset_label[...] = label_train
f.close()

# Save reshaped testing data
f = h5py.File('./test_reshaped.hdf5', 'w')
dataset_input = f.create_dataset('image', (len(input_test), img_width, img_height, img_num_channels))
dataset_label = f.create_dataset('label', (len(input_test),))
dataset_input[...] = input_test
dataset_label[...] = label_test
f.close()
```

#### Full preprocessing code

If you wish to obtain the full code for preprocessing at once - of course, that's possible. Here you go :)

```
import h5py

# Configuration
img_width, img_height, img_num_channels = 28, 28, 1

# Load MNIST data
f = h5py.File('./train.hdf5', 'r')
input_train = f['image'][...]
label_train = f['label'][...]
f.close()
f = h5py.File('./test.hdf5', 'r')
input_test = f['image'][...]
label_test = f['label'][...]
f.close()

# Reshape data
input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
input_test  = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Scale data
input_train = input_train / 255
input_test = input_test / 255

# Save reshaped training data
f = h5py.File('./train_reshaped.hdf5', 'w')
dataset_input = f.create_dataset('image', (len(input_train), img_width, img_height, img_num_channels))
dataset_label = f.create_dataset('label', (len(input_train),))
dataset_input[...] = input_train
dataset_label[...] = label_train
f.close()

# Save reshaped testing data
f = h5py.File('./test_reshaped.hdf5', 'w')
dataset_input = f.create_dataset('image', (len(input_test), img_width, img_height, img_num_channels))
dataset_label = f.create_dataset('label', (len(input_test),))
dataset_input[...] = input_test
dataset_label[...] = label_test
f.close()
```

### Training the model

At this point, we have HDF5 files that we can actually use to train a [Keras based ConvNet](https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/)!

Let's take a look.

Small note: if you wish to understand how to create a ConvNet with Keras `Conv2D` layers, I'd advise you click the link above - as it will take you through all the steps. There's no point repeating them here. Below, we will primarily focus on the `HDF5Matrix` linkage to the Keras model, in order not to confuse ourselves.

#### Imports and model configuration

First, we specify the imports - make sure that `tensorflow` and especially TensorFlow 2.x is installed on your system:

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import HDF5Matrix
```

Then, we set the model configuration:

```
# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 28, 28, 1
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 25
optimizer = Adam()
validation_split = 0.2
verbosity = 1
```

#### Loading data with HDF5Matrix

Now, we can show how the `HDF5Matrix` works:

```
# Load MNIST data
input_train = HDF5Matrix('./train_reshaped.hdf5', 'image')
input_test = HDF5Matrix('./test_reshaped.hdf5', 'image')
label_train = HDF5Matrix('./train_reshaped.hdf5', 'label')
label_test = HDF5Matrix('./test_reshaped.hdf5', 'label')
```

Yep. It's that simple. We assign the output of calling `HDF5Matrix` to arrays, and specify the specific HDF5 dataset that we wish to load :)

#### Model specification, compilation and training

Subsequently, we perform the common steps of model specification, compilation and training a.k.a. fitting the data to the compiled model:

```
# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Display a model summary
model.summary()

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(input_train, label_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, label_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

Et voila, we have a Keras model that can be trained with `HDF5Matrix`! :) Running the model in your ML environment, should indeed yield a training process that starts:

```
Epoch 1/25
2020-04-26 19:44:12.481645: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
2020-04-26 19:44:12.785200: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-04-26 19:44:13.975734: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once.
48000/48000 [==============================] - 13s 274us/sample - loss: 0.1159 - accuracy: 0.9645 - val_loss: 0.0485 - val_accuracy: 0.9854
Epoch 2/25
48000/48000 [======================>
```

#### Full model code

Once again, should you wish to obtain the full model code in order to play straight away - here you go

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import HDF5Matrix

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 28, 28, 1
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 25
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# Load MNIST data
input_train = HDF5Matrix('./train_reshaped.hdf5', 'image')
input_test = HDF5Matrix('./test_reshaped.hdf5', 'image')
label_train = HDF5Matrix('./train_reshaped.hdf5', 'label')
label_test = HDF5Matrix('./test_reshaped.hdf5', 'label')

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Display a model summary
model.summary()

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(input_train, label_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, label_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

* * *

## Summary

In this blog post, we looked at HDF5 data and the Keras `HDF5Matrix` for loading your data from HDF5 file directly. Firstly, we discussed the HDF5 format itself - which stands for Hierarchical Data Format, version 5, and is composed of groups of groups and groups of datasets that together form a hierarchical data structure. It is especially useful for large datasets, as they can be easily transferred (it's as simple as transferring one file) and loaded quite quickly (it's one of the faster formats, especially compared to text based files).

Subsequently, we looked at the HDF5Matrix implementation within the Keras API. It can be used to load datasets from HDF5 format into memory directly, after which you can use them for training your Keras deep learning model.

Finally, we provided an example implementation with TensorFlow 2.x based Keras and Python - to show you how you can do two things:

- Adapt the dataset a priori to using it with `h5py`, as some raw datasets must be reshaped, scaled and cast;
- Training a Keras neural network with the adapted dataset.

That's it for today! :) I hope you've learnt something from this blog post. If you did, please feel free to leave a message in the comments section below 💬👇. Please do the same if you have any questions or remarks - I'll happily answer you. Thank you for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

## References

Keras. (n.d.). _I/O Utils_. Home - Keras Documentation. [https://keras.io/io\_utils/](https://keras.io/io_utils/)

Wikipedia. (2004, May 4). _Hierarchical data format_. Wikipedia, the free encyclopedia. Retrieved April 13, 2020, from [https://en.wikipedia.org/wiki/Hierarchical\_Data\_Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)
