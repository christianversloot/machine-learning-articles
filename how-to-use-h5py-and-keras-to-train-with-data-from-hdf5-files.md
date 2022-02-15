---
title: "How to use H5Py and Keras to train with data from HDF5 files?"
date: "2020-04-13"
categories: 
  - "deep-learning"
  - "frameworks"
tags: 
  - "dataset"
  - "deep-learning"
  - "h5py"
  - "hdf5"
  - "keras"
  - "machine-learning"
  - "mnist"
---

In the many simple educational cases where people show you how to build Keras models, data is often loaded from the [Keras datasets module](https://www.machinecurve.com/index.php/2019/12/31/exploring-the-keras-datasets/) - where loading the data is as simple as adding one line of Python code.

However, it's much more common that data is delivered in the HDF5 file format - and then you might stuck, especially if you're a beginner.

How to use this format for your machine learning model? How can I train a model with data stored in the HDF5 format? That's what we will look at in today's blog post. We'll be studying the Hierarchical Data Format, as the data format is called, as well as how to access such files in Python - with `h5py`. Then, we actually create a Keras model that is trained with MNIST data, but this time not loaded from the Keras Datasets module - but from HDF5 files instead.

Do note that there's also a different way of working with HDF5 files in Keras - being, with the HDF5Matrix util. While this works great, I found it difficult to _adapt data_ when using it. That means, if your dataset already has the correct structure (e.g. my problem was that I wanted to add image channels to 1-channel RGB images stored in HDF5 format, which isn't really possible with HDF5Matrix, as we shall see later here), it's wise to use this util. If not, you can proceed with this blog post. We'll cover the HDF5Matrix in a different one.

Are you ready? Let's go! 😊

* * *

\[toc\]

* * *

## What is an HDF5 file?

You see them every now and then: HDF5 files. Let's see what such a file is before we actually start working with them. If we go to Wikipedia, we see that...

> Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data.
> 
> Wikipedia (2004)

It's a file format that is specifically designed for large datasets. That might be what we need sometimes for our machine learning projects!

Let's now take a slightly closer look at the structure of the HDF format, specifically for HDF5 files - as in my opinion, the HDF4 format is outdated.

It consists of **datasets** and **groups**, where (Wikipedia, 2004)...

- Datasets are multidimensional arrays of a homogeneous type
- Groups are container structures which can hold datasets and other groups.

According to Wikipedia, this creates a truly hierarchical data structure. The multidimensional array structure can hold our data, whereas targets and labels can be split between two different datasets. Finally, the different _classes_ of your dataset, spread between two datasets per class (target / label), can be structured into multiple groups.

A very handy format indeed!

https://www.youtube.com/watch?v=q14F3WRwSck

* * *

## Why use HDF5 instead of CSV/text when storing ML datasets?

There is a wide range of possible file types which you can use to store data. HDF5 is one example, but you could also use SQL based solutions like SQLite, or plain text files / CSVs. However, if we take a look at a post by Alex I. (n.d.), HDF5 has some advantages over these data types:

1. While databases can be an advantage in terms of data that cannot be stored in memory, they are often slower than HDF5 files. You must make this trade-off depending on the size of your dataset.
2. The same goes for text files. While they can be "fairly space-efficient" (especially when compressed substantially), they are slower to use as "parsing text is much, much slower than HDF".
3. While "other binary formats" like Numpy arrays are quite good, they are not as widely supported as HDF, which is the "lingua franca or common interchange format".

The author also reports that whereas "a certain small dataset" took 2 seconds to read as HDF, 1 minute to read as JSON, and 1 hour to write to database.

You get the point :)

* * *

## A Keras example

Now, let's take a look if we can create a simple [Convolutional Neural Network](https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/) which operates with the [MNIST dataset](https://www.machinecurve.com/index.php/2019/12/31/exploring-the-keras-datasets/#mnist-database-of-handwritten-digits), stored in HDF5 format.

Fortunately, this dataset is readily available at [Kaggle for download](https://www.kaggle.com/benedictwilkinsai/mnist-hd5f), so make sure to create an account there and download the **train.hdf5** and **test.hdf5** files.

### The differences: the imports & how to load the data

Our HDF5 based model is not too different compared to any other Keras model. In fact, the only differences are present at the start - namely, an extra import as well as a different way of loading the data. That's what we'll highlight in this post primarily. If you wish to understand the ConvNet creation process in more detail, I suggest you also take a look at [this blog](https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/).

### The imports

The imports first. The only thing that we will add to the imports we already copied from that other blog is the `import h5py` statement:

```
import h5py
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
```

This is what H5py does:

> **HDF5 for Python**  
> The h5py package is a Pythonic interface to the HDF5 binary data format.
> 
> H5py (n.d.)

We can thus use it to access the data, which we'll do now.

### Loading the data

Let's put the model configuration in your file next:

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

Followed by loading and reshaping the input data into the correct [input shape](https://www.machinecurve.com/index.php/2020/04/05/how-to-find-the-value-for-keras-input_shape-input_dim/) (i.e. _length_ of the datasets times `(28, 28, 1)` as MNIST contains grayscale 28x28 pixels images). Here's the code for that:

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

# Reshape data
input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
input_test  = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))
```

...interpreting it is actually pretty simple. We use `h5py` to load the two HDF5 files, one with the training data, the other with the testing data.

From the HDF5 files, we retrieve the `image` and `label` datasets, where the `[...]` indicates that we retrieve every individual sample - which means 60.000 samples in the training case, for example.

Don't forget to close the files once you've finished working with them, before starting the reshaping process.

That's pretty much it with respect to loading data from HDF5!

### Full model code

We can now add the other code which creates, configures and trains the Keras model, which means that we end with this code as a whole:

```
import h5py
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

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

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

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

### Let's run it

Now, save this model - e.g. as `h5model.py` - and open a terminal. `cd` to the folder where your file is located and execute it with `python h5model.py`.

Make sure that TensorFlow 2.x is installed, as well as `h5py`:

- [Installing TensorFlow 2.x onto your system](https://www.tensorflow.org/install);
- [Installing H5py onto your system](http://docs.h5py.org/en/stable/build.html).

Then, you should see the training process begin - as we are used to:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 22, 22, 128)       73856
_________________________________________________________________
flatten (Flatten)            (None, 61952)             0
_________________________________________________________________
dense (Dense)                (None, 128)               7929984
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 8,023,946
Trainable params: 8,023,946
Non-trainable params: 0
_________________________________________________________________
Train on 48000 samples, validate on 12000 samples
Epoch 1/25
2020-04-13 15:15:25.949751: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
2020-04-13 15:15:26.217503: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-04-13 15:15:27.236616: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once.
48000/48000 [=========================
```

We've done the job! 😊

* * *

## Summary

In this blog post, we answered the question _how to use datasets represented in HDF5 files for training your Keras model?_ Despite the blog being relatively brief, I think that it helps understanding what HDF5 is, how we can use it in Python through h5py, and how we can subsequently prepare the HDF5-loaded data for training your Keras model.

Hopefully, you've learnt something new today! If you did, I'd appreciate a comment - please feel free to leave one in the comments section below. Please do the same if you have any questions or other remarks. In any case, thank you for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

## References

Wikipedia. (2004, May 4). _Hierarchical data format_. Wikipedia, the free encyclopedia. Retrieved April 13, 2020, from [https://en.wikipedia.org/wiki/Hierarchical\_Data\_Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)

Alex I. (n.d.). _Hierarchical data format. What are the advantages compared to alternative formats?_ Data Science Stack Exchange. [https://datascience.stackexchange.com/a/293](https://datascience.stackexchange.com/a/293)

BenedictWilkinsAI. (n.d.). _Mnist - Hdf5_. Kaggle: Your Machine Learning and Data Science Community. [https://www.kaggle.com/benedictwilkinsai/mnist-hd5f](https://www.kaggle.com/benedictwilkinsai/mnist-hd5f)

H5py. (n.d.). _HDF5 for Python — h5py 2.10.0 documentation_. [https://docs.h5py.org/en/stable/index.html](https://docs.h5py.org/en/stable/index.html)
