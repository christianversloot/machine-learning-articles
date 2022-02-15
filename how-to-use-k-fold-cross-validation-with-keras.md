---
title: "How to use K-fold Cross Validation with TensorFlow 2 and Keras?"
date: "2020-02-18"
categories: 
  - "buffer"
  - "frameworks"
  - "svms"
tags: 
  - "dataset"
  - "k-fold-cross-validation"
  - "split"
  - "training-process"
  - "training-split"
  - "validation"
---

When you train supervised machine learning models, you'll likely try multiple models, in order to find out how good they are. Part of this process is likely going to be the question _how can I compare models objectively?_

Training and testing datasets have been invented for this purpose. By splitting a small part off your full dataset, you create a dataset which (1) was not yet seen by the model, and which (2) you assume to approximate the distribution of the _population_, i.e. the real world scenario you wish to generate a predictive model for.

Now, when generating such a split, you should ensure that your splits are relatively unbiased. In this blog post, we'll cover one technique for doing so: **K-fold Cross Validation**. Firstly, we'll show you how such splits can be made naïvely - i.e., by a simple hold out split strategy. Then, we introduce K-fold Cross Validation, show you how it works, and why it can produce better results. This is followed by an example, created with Keras and Scikit-learn's KFold functions.

Are you ready? Let's go! 😎

[Ask a question](https://www.machinecurve.com/index.php/add-machine-learning-question/)

* * *

**Update 12/Feb/2021:** added TensorFlow 2 to title; some styling changes.

**Update 11/Jan/2021:** added code example to start using K-fold CV straight away.

**Update 04/Aug/2020:** clarified the (in my view) necessity of validation set even after K-fold CV.

**Update 11/Jun/2020:** improved K-fold cross validation code based on reader comments.

* * *

\[toc\]

* * *

## Code example: K-fold Cross Validation with TensorFlow and Keras

This quick code can be used to perform K-fold Cross Validation with your TensorFlow/Keras model straight away. If you want to understand it in more detail, make sure to read the rest of the article below!

```
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1
```

* * *

## Evaluating and selecting models with K-fold Cross Validation

Training a [supervised machine learning model](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process) involves changing model weights using a _training set_. Later, once training has finished, the trained model is tested with new data - the _testing set_ - in order to find out how well it performs in real life.

When you are satisfied with the performance of the model, you train it again with the entire dataset, in order to finalize it and use it in production (Bogdanovist, n.d.)

However, when checking how well the model performance, the question _how to split the dataset_ is one that emerges pretty rapidly. K-fold Cross Validation, the topic of today's blog post, is one possible approach, which we'll discuss next.

However, let's first take a look at the concept of generating train/test splits in the first place. Why do you need them? Why can't you simply train the model with all your data and then compare the results with other models? We'll answer these questions first.

Then, we take a look at the efficient but naïve _simple hold-out splits_. This way, when we discuss K-fold Cross Validation, you'll understand more easily why it can be more useful when comparing performance between models. Let's go!

### Why using train/test splits? - On finding a model that works for you

Before we'll dive into the approaches for generating train/test splits, I think that it's important to take a look at _why we should split them_ in the first place when evaluating model performance.

For this reason, we'll invent a model evaluation scenario first.

#### Generating many predictions

Say that we're training a few models to classify images of digits. We train a [Support Vector Machine](https://www.machinecurve.com/index.php/2019/09/20/intuitively-understanding-svm-and-svr/) (SVM), a [Convolutional Neural Network](https://www.machinecurve.com/index.php/2018/12/07/convolutional-neural-networks-and-their-components-for-computer-vision/) (CNN) and a [Densely-connected Neural Network](https://www.machinecurve.com/index.php/2019/07/27/how-to-create-a-basic-mlp-classifier-with-the-keras-sequential-api/) (DNN) and of course, hope that each of them predicts "5" in this scenario:

[![](images/EvaluationScenario-1024x366.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/EvaluationScenario.png)

Our goal here is to use the model that performs best in production, a.k.a. "really using it" :)

The central question then becomes: **how well does each model perform?**

Based on their performance, we can select a model that can be used in real life.

However, if we wish to determine model performance, we should generate a whole bunch of predictions - preferably, thousands or even more - so that we can compute metrics like accuracy, or [loss](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/). Great!

#### Don't be the student who checks his own homework

Now, we'll get to the core of our point - i.e., why we need to generate splits between training and testing data when evaluating machine learning models.

We'll require an understanding of the high-level supervised machine learning process for this purpose:

[![](images/High-level-training-process-1024x973.jpg)](https://www.machinecurve.com/wp-content/uploads/2019/09/High-level-training-process.jpg)

It can be read as follows:

- In the first step, all the training samples (in blue on the left) are fed forward to the machine learning model, which generates predictions (blue on the right).
- In the second step, the predictions are compared with the "ground truth" (the real targets) - which results in the computation of a [loss value](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/).
- The model can subsequently be optimized by steering the model away from the error, by changing its weights, in the backwards pass of the gradient with respect to (finally) the loss value.
- The process then starts again. Presumably, the model performs better this time.

As you can imagine, the model will improve based on the _loss generated by the data_. This data is a _sample_, which means that there is always a difference between the _sample distribution_ and the _population distribution_. In other words, there is always a difference between _what your data tells that the patterns are_ and _what the patterns are in the real world_. This difference can be really small, but it's there.

Now, if you let the model train for long enough, it will adapt substantially to the dataset. This also means that the impact of the difference will get larger and larger, relative to the patterns of the real-world scenario. If you've trained it for too long - [a problem called overfitting](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/) - the difference may be the cause that it won't work anymore when real world data is fed to it.

Generating a split between training data and testing data can help you solve this issue. By training your model using the training data, you can let it train for as long as you want. Why? Simple: you have the testing data to evaluate model performance afterwards, using data that is (1) presumably representative for the real world and (2) unseen yet. If the model is highly overfit, this will be clear, because it will perform very poorly during the evaluation step with the testing data.

Now, let's take a look at how we can do this. We'll s tart with simple hold-out splits :)

### A naïve approach: simple hold-out split

Say that you've got a dataset of 10.000 samples. It hasn't been split into a training and a testing set yet. Generally speaking, a 80/20 split is acceptable. That is, 80% of your data - 8.000 samples in our case - will be used for training purposes, while 20% - 2.000 - will be used for testing.

We can thus simply draw a boundary at 8.000 samples, like this:

[![](images/Traintest.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/Traintest.png)

We call this _simple hold-out split_, as we simply "hold out" the last 2.000 samples (Chollet, 2017).

It can be a highly effective approach. What's more, it's also very inexpensive in terms of the computational power you need. However, it's also a very naïve approach, as you'll have to keep these edge cases in mind all the time (Chollet, 2017):

1. **Data representativeness**: all datasets, which are essentially samples, must represent the patterns in the population as much as possible. This becomes especially important when you generate samples from a sample (i.e., from your full dataset). For example, if the first part of your dataset has pictures of ice cream, while the latter one only represents espressos, trouble is guaranteed when you generate the split as displayed above. Random shuffling may help you solve these issues.
2. **The arrow of time**: if you have a time series dataset, your dataset is likely ordered chronologically. If you'd shuffle randomly, and then perform simple hold-out validation, you'd effectively "\[predict\] the future given the past" (Chollet, 2017). Such temporal leaks don't benefit model performance.
3. **Data redundancy**: if some samples appear more than once, a simple hold-out split with random shuffling may introduce redundancy between training and testing datasets. That is, identical samples belong to both datasets. This is problematic too, as data used for training thus leaks into the dataset for testing implicitly.

Now, as we can see, while a simple hold-out split based approach can be effective and will be efficient in terms of computational resources, it also requires you to monitor for these edge cases continuously.

\[affiliatebox\]

### K-fold Cross Validation

A more expensive and less naïve approach would be to perform K-fold Cross Validation. Here, you set some value for \[latex\]K\[/latex\] and (hey, what's in a name 😋) the dataset is split into \[latex\]K\[/latex\] partitions of equal size. \[latex\]K - 1\[/latex\] are used for training, while one is used for testing. This process is repeated \[latex\]K\[/latex\] times, with a different partition used for testing each time.

For example, this would be the scenario for our dataset with \[latex\]K = 5\[/latex\] (i.e., once again the 80/20 split, but then 5 times!):

[![](images/KTraintest.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/KTraintest.png)

For each split, the same model is trained, and performance is displayed per fold. For evaluation purposes, you can obviously also average it across all folds. While this produces better estimates, K-fold Cross Validation also increases training cost: in the \[latex\]K = 5\[/latex\] scenario above, the model must be trained for 5 times.

Let's now extend our viewpoint with a few variations of K-fold Cross Validation :)

If you have no computational limitations whatsoever, you might wish to try a special case of K-fold Cross Validation, called Leave One Out Cross Validation (or LOOCV, Khandelwal 2019). LOOCV means \[latex\]K = N\[/latex\], where \[latex\]N\[/latex\] is the number of samples in your dataset. As the number of models trained is maximized, the precision of the model performance average is maximized too, but so is the cost of training due to the sheer amount of models that must be trained.

If you have a binary classification problem, you might also wish to take a look at Stratified Cross Validation (Khandelwal, 2019). It extends K-fold Cross Validation by ensuring an equal distribution of the target classes over the splits. This ensures that your classification problem is balanced. It doesn't work for multiclass classification due to the way that samples are distributed.

Finally, if you have a time series dataset, you might wish to use Time-series Cross Validation (Khandelwal, 2019). [Check here how it works.](https://medium.com/datadriveninvestor/k-fold-and-other-cross-validation-techniques-6c03a2563f1e#4a74)

* * *

## Creating a Keras model with K-fold Cross Validation

Now that we understand how K-fold Cross Validation works, it's time to code an example with the Keras deep learning framework :)

Coding it will be a multi-stage process:

- Firstly, we'll take a look at what we need in order to run our model successfully.
- Then, we take a look at today's model.
- Subsequently, we add K-fold Cross Validation, train the model instances, and average performance.
- Finally, we output the performance metrics on screen.

### What we'll need to run our model

For running the model, we'll need to install a set of software dependencies. For today's blog post, they are as follows:

- TensorFlow 2.0+, which includes the Keras deep learning framework;
- The most recent version of scikit-learn;
- Numpy.

That's it, already! :)

### Our model: a CIFAR-10 CNN classifier

Now, today's model.

We'll be using a [convolutional neural network](https://www.machinecurve.com/index.php/2018/12/07/convolutional-neural-networks-and-their-components-for-computer-vision/) that can be used to classify CIFAR-10 images into a set of 10 classes. The images are varied, as you can see here:

[![](images/cifar10_images.png)](https://www.machinecurve.com/wp-content/uploads/2019/11/cifar10_images.png)

Now, my goal is not to replicate the process of creating the model here, as we already did that in our blog post ["How to build a ConvNet for CIFAR-10 and CIFAR-100 classification with Keras?"](https://www.machinecurve.com/index.php/2020/02/09/how-to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras/). Take a look at that post if you wish to understand the steps that lead to the model below.

_(Do note that this is a small adaptation, where we removed the third convolutional block for reasons of speed.)_

Here is the full model code of the original CIFAR-10 CNN classifier, which we can use when adding K-fold Cross Validation:

```
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 100
no_epochs = 100
optimizer = Adam()
verbosity = 1

# Load CIFAR-10 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
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
            verbose=verbosity)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()
```

### Removing obsolete code

Now, let's slightly adapt the model in order to add K-fold Cross Validation.

Firstly, we'll strip off some code that we no longer need:

```
import matplotlib.pyplot as plt
```

We will no longer generate the visualizations, and besides the import we thus also remove the part generating them:

```
# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()
```

### Adding K-fold Cross Validation

Secondly, let's add the `KFold` code from `scikit-learn` to the imports - as well as `numpy`:

```
from sklearn.model_selection import KFold
import numpy as np
```

Which...

> Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
> 
> Scikit-learn (n.d.) [sklearn.model\_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

Precisely what we want!

We also add a new configuration value:

```
num_folds = 10
```

This will ensure that our \[latex\]K = 10\[/latex\].

What's more, directly after the "normalize data" step, we add two empty lists for storing the results of cross validation:

```
# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []
```

This is followed by a concat of our 'training' and 'testing' datasets - remember that K-fold Cross Validation makes the split!

```
# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)
```

Based on this prior work, we can add the code for K-fold Cross Validation:

```
fold_no = 1
for train, test in kfold.split(input_train, target_train):
```

Ensure that all the `model` related steps are now wrapped inside the `for` loop. Also make sure to add a couple of extra `print` statements and to replace the inputs and targets to `model.fit`:

```
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)
```

We next replace the "test loss" `print` with one related to what we're doing. Also, we increase the `fold_no`:

```
  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1
```

Here, we simply print a "score for fold X" - and add the accuracy and sparse categorical crossentropy loss values to the lists.

Now, why do we do that?

Simple: at the end, we provide an overview of all scores and the averages. This allows us to easily compare the model with others, as we can simply compare these outputs. Add this code at the end of the model, but make sure that it is _not_ wrapped inside the `for` loop:

```
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
```

#### Full model code

Altogether, this is the new code for your K-fold Cross Validation scenario with \[latex\]K = 10\[/latex\]:

\[affiliatebox\]

```
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 100
no_epochs = 25
optimizer = Adam()
verbosity = 1
num_folds = 10

# Load CIFAR-10 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
```

* * *

## Results

Now, it's time to run the model, to see whether we can get some nice results :)

Say, for example, that you saved the model as `k-fold-model.py` in some folder. Open up your command prompt - for example, Anaconda Prompt - and `cd` to the folder where your file is stored. Make sure that your dependencies are installed and then run `python k-fold-model.py`.

If everything goes well, the model should start training for 25 epochs per fold.

### Evaluating the performance of your model

During training, it should produce batches like this one:

```
------------------------------------------------------------------------
Training for fold 3 ...
Train on 43200 samples, validate on 10800 samples
Epoch 1/25
43200/43200 [==============================] - 9s 200us/sample - loss: 1.5628 - accuracy: 0.4281 - val_loss: 1.2300 - val_accuracy: 0.5618
Epoch 2/25
43200/43200 [==============================] - 7s 165us/sample - loss: 1.1368 - accuracy: 0.5959 - val_loss: 1.0767 - val_accuracy: 0.6187
Epoch 3/25
43200/43200 [==============================] - 7s 161us/sample - loss: 0.9737 - accuracy: 0.6557 - val_loss: 0.9869 - val_accuracy: 0.6522
Epoch 4/25
43200/43200 [==============================] - 7s 169us/sample - loss: 0.8665 - accuracy: 0.6967 - val_loss: 0.9347 - val_accuracy: 0.6772
Epoch 5/25
43200/43200 [==============================] - 8s 175us/sample - loss: 0.7792 - accuracy: 0.7281 - val_loss: 0.8909 - val_accuracy: 0.6918
Epoch 6/25
43200/43200 [==============================] - 7s 168us/sample - loss: 0.7110 - accuracy: 0.7508 - val_loss: 0.9058 - val_accuracy: 0.6917
Epoch 7/25
43200/43200 [==============================] - 7s 161us/sample - loss: 0.6460 - accuracy: 0.7745 - val_loss: 0.9357 - val_accuracy: 0.6892
Epoch 8/25
43200/43200 [==============================] - 8s 184us/sample - loss: 0.5885 - accuracy: 0.7963 - val_loss: 0.9242 - val_accuracy: 0.6962
Epoch 9/25
43200/43200 [==============================] - 7s 156us/sample - loss: 0.5293 - accuracy: 0.8134 - val_loss: 0.9631 - val_accuracy: 0.6892
Epoch 10/25
43200/43200 [==============================] - 7s 164us/sample - loss: 0.4722 - accuracy: 0.8346 - val_loss: 0.9965 - val_accuracy: 0.6931
Epoch 11/25
43200/43200 [==============================] - 7s 161us/sample - loss: 0.4168 - accuracy: 0.8530 - val_loss: 1.0481 - val_accuracy: 0.6957
Epoch 12/25
43200/43200 [==============================] - 7s 159us/sample - loss: 0.3680 - accuracy: 0.8689 - val_loss: 1.1481 - val_accuracy: 0.6938
Epoch 13/25
43200/43200 [==============================] - 7s 165us/sample - loss: 0.3279 - accuracy: 0.8850 - val_loss: 1.1438 - val_accuracy: 0.6940
Epoch 14/25
43200/43200 [==============================] - 7s 171us/sample - loss: 0.2822 - accuracy: 0.8997 - val_loss: 1.2441 - val_accuracy: 0.6832
Epoch 15/25
43200/43200 [==============================] - 7s 167us/sample - loss: 0.2415 - accuracy: 0.9149 - val_loss: 1.3760 - val_accuracy: 0.6786
Epoch 16/25
43200/43200 [==============================] - 7s 170us/sample - loss: 0.2029 - accuracy: 0.9294 - val_loss: 1.4653 - val_accuracy: 0.6820
Epoch 17/25
43200/43200 [==============================] - 7s 165us/sample - loss: 0.1858 - accuracy: 0.9339 - val_loss: 1.6131 - val_accuracy: 0.6793
Epoch 18/25
43200/43200 [==============================] - 7s 171us/sample - loss: 0.1593 - accuracy: 0.9439 - val_loss: 1.7192 - val_accuracy: 0.6703
Epoch 19/25
43200/43200 [==============================] - 7s 168us/sample - loss: 0.1271 - accuracy: 0.9565 - val_loss: 1.7989 - val_accuracy: 0.6807
Epoch 20/25
43200/43200 [==============================] - 8s 190us/sample - loss: 0.1264 - accuracy: 0.9547 - val_loss: 1.9215 - val_accuracy: 0.6743
Epoch 21/25
43200/43200 [==============================] - 9s 207us/sample - loss: 0.1148 - accuracy: 0.9587 - val_loss: 1.9823 - val_accuracy: 0.6720
Epoch 22/25
43200/43200 [==============================] - 7s 167us/sample - loss: 0.1110 - accuracy: 0.9615 - val_loss: 2.0952 - val_accuracy: 0.6681
Epoch 23/25
43200/43200 [==============================] - 7s 166us/sample - loss: 0.0984 - accuracy: 0.9653 - val_loss: 2.1623 - val_accuracy: 0.6746
Epoch 24/25
43200/43200 [==============================] - 7s 168us/sample - loss: 0.0886 - accuracy: 0.9691 - val_loss: 2.2377 - val_accuracy: 0.6772
Epoch 25/25
43200/43200 [==============================] - 7s 166us/sample - loss: 0.0855 - accuracy: 0.9697 - val_loss: 2.3857 - val_accuracy: 0.6670
Score for fold 3: loss of 2.4695983460744224; accuracy of 66.46666526794434%
------------------------------------------------------------------------
```

Do note the increasing validation loss, a clear [sign of overfitting](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/).

And finally, after the 10th fold, it should display the overview with results per fold and the average:

```
------------------------------------------------------------------------
Score per fold
------------------------------------------------------------------------
> Fold 1 - Loss: 2.4094747734069824 - Accuracy: 67.96666383743286%
------------------------------------------------------------------------
> Fold 2 - Loss: 1.768296229839325 - Accuracy: 67.03333258628845%
------------------------------------------------------------------------
> Fold 3 - Loss: 2.4695983460744224 - Accuracy: 66.46666526794434%
------------------------------------------------------------------------
> Fold 4 - Loss: 2.363724467277527 - Accuracy: 66.28333330154419%
------------------------------------------------------------------------
> Fold 5 - Loss: 2.083754387060801 - Accuracy: 65.51666855812073%
------------------------------------------------------------------------
> Fold 6 - Loss: 2.2160572570165 - Accuracy: 65.6499981880188%
------------------------------------------------------------------------
> Fold 7 - Loss: 1.7227793588638305 - Accuracy: 66.76666736602783%
------------------------------------------------------------------------
> Fold 8 - Loss: 2.357142448425293 - Accuracy: 67.25000143051147%
------------------------------------------------------------------------
> Fold 9 - Loss: 1.553109979470571 - Accuracy: 65.54999947547913%
------------------------------------------------------------------------
> Fold 10 - Loss: 2.426255855560303 - Accuracy: 66.03333353996277%
------------------------------------------------------------------------
Average scores for all folds:
> Accuracy: 66.45166635513306 (+- 0.7683473645622098)
> Loss: 2.1370193102995554
------------------------------------------------------------------------
```

This allows you to compare the performance across folds, and compare the averages of the folds across model types you're evaluating :)

In our case, the model produces accuracies of 60-70%. This is acceptable, but there is still room for improvement. But hey, that wasn't the scope of this blog post :)

### Model finalization

If you're satisfied with the performance of your model, you can _finalize_ it. There are two options for doing so:

- Save the best performing model instance (check ["How to save and load a model with Keras?"](https://www.machinecurve.com/index.php/2020/02/14/how-to-save-and-load-a-model-with-keras/) - do note that this requires retraining because you haven't saved models with the code above), and use it for generating predictions.
- Retrain the model, but this time with all the data - i.e., without making the train/test split. Save that model, and use it for generating predictions. I do suggest to continue using a validation set, as you want to know when the model [is overfitting](https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/).

Both sides have advantages and disadvantages. The advantages of the first are that you don't have to retrain, as you can simply use the best-performing fold which was saved _during_ the training procedure. As retraining may be expensive, this could be an option, especially when your model is large. However, the disadvantage is that you simply miss out a percentage of your data - which may bring your training sample closer to the actual patterns in the _population_ rather than your _sample_. If that's the case, then the second option is better.

However, that's entirely up to you! :)

* * *

## Summary

In this blog post, we looked at the concept of model evaluation: what is it? Why would we need it in the first place? And how to do so objectively? If we can't evaluate models without introducing bias of some sort, there's no point in evaluating at all, is there?

We introduced simple hold-out splits for this purpose, and showed that while they are efficient in terms of the required computational resources, they are also naïve. K-fold Cross Validation is \[latex\]K\[/latex\] times more expensive, but can produce significantly better estimates because it trains the models for \[latex\]K\[/latex\] times, each time with a different train/test split.

To illustrate this further, we provided an example implementation for the Keras deep learning framework using TensorFlow 2.0. Using a Convolutional Neural Network for CIFAR-10 classification, we generated evaluations that performed in the range of 60-70% accuracies.

[Ask a question](https://www.machinecurve.com/index.php/add-machine-learning-question/)

I hope you've learnt something from today's blog post. If you did, feel free to leave a comment in the comments section! If you have questions, you can add a comment or ask a question with the button on the right. Please do the same if you spotted mistakes or when you have other remarks. I'll happily answer your comments and will improve my blog if that's the best thing to do.

Thank you for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

* * *

## References

Scikit-learn. (n.d.). sklearn.model\_selection.KFold — scikit-learn 0.22.1 documentation. Retrieved February 17, 2020, from [https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.KFold.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

Allibhai, E. (2018, October 3). Holdout vs. Cross-validation in Machine Learning. Retrieved from [https://medium.com/@eijaz/holdout-vs-cross-validation-in-machine-learning-7637112d3f8f](https://medium.com/@eijaz/holdout-vs-cross-validation-in-machine-learning-7637112d3f8f)

Chollet, F. (2017). _Deep Learning with Python_. New York, NY: Manning Publications.

Khandelwal, R. (2019, January 25). K fold and other cross-validation techniques. Retrieved from [https://medium.com/datadriveninvestor/k-fold-and-other-cross-validation-techniques-6c03a2563f1e](https://medium.com/datadriveninvestor/k-fold-and-other-cross-validation-techniques-6c03a2563f1e)

Bogdanovist. (n.d.). How to choose a predictive model after k-fold cross-validation? Retrieved from [https://stats.stackexchange.com/a/52277](https://stats.stackexchange.com/a/52277)

* * *
