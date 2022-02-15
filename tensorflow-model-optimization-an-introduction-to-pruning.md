---
title: "TensorFlow model optimization: an introduction to Pruning"
date: "2020-09-23"
categories: 
  - "frameworks"
tags: 
  - "edge-ai"
  - "optimizer"
  - "pruning"
  - "quantization"
  - "tensorflow"
  - "model-optimization"
---

Enjoying the benefits of machine learning models means that they are deployed in the field after training has finished. However, if you're counting on great speed with which predictions for new data - called model inference - are generated, then it's possible that you're getting a bit intimidated. If you _really_ want your models to run with speed, it's likely that you'll have to buy powerful equipment - like massive GPUs - which come at significant cost.

If you don't, your models will run slower; sometimes, really slow - especially when your models are big. And big models are very common in today's state-of-the-art in machine learning.

Fortunately, modern machine learning frameworks such as TensorFlow attempt to help machine learning engineers. Through extensions such as TF Lite, methods such as [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/) can be used to optimize your model. While with quantization the number representation of your machine learning model is adapted to benefit size and speed (often at the cost of precision), we'll take a look at **model pruning** in this article. Firstly, we'll take a look at why model optimization is necessary. Subsequently, we'll introduce pruning - by taking a look at how neural networks work as well as questioning why we should keep weights that don't contribute to model performance.

Following the theoretical part of this article, we'll build a Keras model and subsequently apply pruning to optimize it. This shows you how to apply pruning to your TensorFlow/Keras model with a real example. Finally, when we know how to do is, we'll continue by _combining_ pruning with quantization for compound optimization. Obviously, this also includes adding quantization to the Keras example that we created before.

Are you ready? Let's go! 😎

**Update 02/Oct/2020:** added reference to article about pruning schedules as a suggestion.

* * *

\[toc\]

* * *

## The need for model optimization

Machine learning models can be used for a wide variety of use cases, for example the detection of objects:

https://www.youtube.com/watch?v=\_zZe27JYi8Y

If you're into object detection, it's likely that you have heard about machine learning architectures like RCNN, Faster-RCNN, YOLO (recently, version 5 was released!) and others. Those are increasingly state-of-the-art architectures that can be used to detect objects very efficiently based on a training dataset.

The architectures are composed of a pipeline that includes a feature extraction model, region proposal network, and subsequently a classification model (Data Science Stack Exchange, n.d.). By consequence, this pipeline is capable of extracting interesting features from your input data, detecting regions of interest for classification, and finally classifying those regions - resulting in videos like the one above.

Now, while they are very performant in terms of object detection, the neural networks used for classifying (and sometimes also for feature extraction/region selection) also come at a downside: **_they are very big_.**

For example, the neural nets, which can include [VGG-16](https://neurohive.io/en/popular-networks/vgg16/), [RESNET-50](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33), and others, have the following size when used as a `tf.keras` application (for example, as a convolutional base):

| Model | Size | Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth |
| --- | --- | --- | --- | --- | --- |
| [Xception](https://keras.io/api/applications/xception) | 88 MB | 0.790 | 0.945 | 22,910,480 | 126 |
| [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) | 528 MB | 0.713 | 0.901 | 138,357,544 | 23 |
| [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) | 549 MB | 0.713 | 0.900 | 143,667,240 | 26 |
| [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function) | 98 MB | 0.749 | 0.921 | 25,636,712 | \- |
| [ResNet101](https://keras.io/api/applications/resnet/#resnet101-function) | 171 MB | 0.764 | 0.928 | 44,707,176 | \- |
| [ResNet152](https://keras.io/api/applications/resnet/#resnet152-function) | 232 MB | 0.766 | 0.931 | 60,419,944 | \- |
| [ResNet50V2](https://keras.io/api/applications/resnet/#resnet50v2-function) | 98 MB | 0.760 | 0.930 | 25,613,800 | \- |
| [ResNet101V2](https://keras.io/api/applications/resnet/#resnet101v2-function) | 171 MB | 0.772 | 0.938 | 44,675,560 | \- |
| [ResNet152V2](https://keras.io/api/applications/resnet/#resnet152v2-function) | 232 MB | 0.780 | 0.942 | 60,380,648 | \- |
| [InceptionV3](https://keras.io/api/applications/inceptionv3) | 92 MB | 0.779 | 0.937 | 23,851,784 | 159 |
| [InceptionResNetV2](https://keras.io/api/applications/inceptionresnetv2) | 215 MB | 0.803 | 0.953 | 55,873,736 | 572 |
| [MobileNet](https://keras.io/api/applications/mobilenet) | 16 MB | 0.704 | 0.895 | 4,253,864 | 88 |
| [MobileNetV2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) | 14 MB | 0.713 | 0.901 | 3,538,984 | 88 |
| [DenseNet121](https://keras.io/api/applications/densenet/#densenet121-function) | 33 MB | 0.750 | 0.923 | 8,062,504 | 121 |
| [DenseNet169](https://keras.io/api/applications/densenet/#densenet169-function) | 57 MB | 0.762 | 0.932 | 14,307,880 | 169 |
| [DenseNet201](https://keras.io/api/applications/densenet/#densenet201-function) | 80 MB | 0.773 | 0.936 | 20,242,984 | 201 |
| [NASNetMobile](https://keras.io/api/applications/nasnet/#nasnetmobile-function) | 23 MB | 0.744 | 0.919 | 5,326,716 | \- |
| [NASNetLarge](https://keras.io/api/applications/nasnet/#nasnetlarge-function) | 343 MB | 0.825 | 0.960 | 88,949,818 | \- |
| [EfficientNetB0](https://keras.io/api/applications/efficientnet/#efficientnetb0-function) | 29 MB | \- | \- | 5,330,571 | \- |
| [EfficientNetB1](https://keras.io/api/applications/efficientnet/#efficientnetb1-function) | 31 MB | \- | \- | 7,856,239 | \- |
| [EfficientNetB2](https://keras.io/api/applications/efficientnet/#efficientnetb2-function) | 36 MB | \- | \- | 9,177,569 | \- |
| [EfficientNetB3](https://keras.io/api/applications/efficientnet/#efficientnetb3-function) | 48 MB | \- | \- | 12,320,535 | \- |
| [EfficientNetB4](https://keras.io/api/applications/efficientnet/#efficientnetb4-function) | 75 MB | \- | \- | 19,466,823 | \- |
| [EfficientNetB5](https://keras.io/api/applications/efficientnet/#efficientnetb5-function) | 118 MB | \- | \- | 30,562,527 | \- |
| [EfficientNetB6](https://keras.io/api/applications/efficientnet/#efficientnetb6-function) | 166 MB | \- | \- | 43,265,143 | \- |
| [EfficientNetB7](https://keras.io/api/applications/efficientnet/#efficientnetb7-function) | 256 MB | \- | \- | 66,658,687 | \- |

Source: Keras Team (n.d.)

Some are approximately half a gigabyte with more than 100 million trainable parameters. That's _really_ big!

The consequences of using those models is that you'll need very powerful hardware in order to perform what is known as **model inference** - or generating new predictions for new data that is input to the trained model. This is why most machine learning settings are centralized and often cloud-based: cloud vendors such as Amazon Web Services, Azure and [DigitalOcean](https://m.do.co/c/2cbc2c399ad5) _(affiliate link)_ provide GPU-based or heavy compute-based machines for running machine learning inference.

Now, this is good if your predictions can be batch oriented or when some delay is acceptable - but if you want to respond to observations in the field, with a very small delay between observation and a response - this is unacceptable.

Very large models, however, cannot run in the field, for the simple reason that insufficiently powerful hardware is available in the field. Embedded devices simply aren't good enough to equal performance of their cloud-based competitors. This means that you'll have to trade-off model performance by using smaller ones.

Fortunately, modern deep learning frameworks provide a variety of techniques to optimize your machine learning models. As we have seen in another blog post, changing the number representation into a less-precise but smaller variant - a technique called [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/) - helps already. In this blog post, we'll take a look at another technique: **model pruning**. Really interesting, especially if you combine the two - as we shall do later! :)

* * *

## Introducing Pruning

Adapting a definition found at Wikipedia (2006) for decision trees, pruning in general means "simplifying/compressing and optimizing a \[classifier\] by removing sections of the \[classifier\] that are uncritical and redundant to classify instances" (Wikipedia, 2006). Hence, while with quantization models are optimized by changing their number representation, pruning allows you to optimize models by removing parts that don't contribute much to the outcome.

I can imagine that it's difficult to visualize this if you don't fully understand how neural networks operate from the inside. Therefore, let's take a look at how they do before we continue introducing pruning.

### Neural network maths: features and weights

Taken from our blog post about loss and loss functions, we can sketch a [high-level machine learning process](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process) for supervised learning scenarios - such as training a classifier or a regression model:

![](images/High-level-training-process-1024x973.jpg)

Training such a model involves a cyclical process, where **features** (or data inputs) are fed to a machine learning model that is initially [initialized](https://www.machinecurve.com/index.php/2019/08/22/what-is-weight-initialization/) quite randomly, after which predictions are compared with the actual outcomes - or the ground truth. After comparison, the model is adapted, after which the process restarts. This way, models are improved incrementally, and "learning" takes place.

If we talk about initializing a machine learning model, we're talking about initializing their **weights**. Each machine learning model has a large amount of weights that can be trained, i.e., where learning can be captured. Both weights and features are vectors. Upon the forward pass (i.e., passing a feature, generating a prediction), the inputs for every layer are fed to the weights, after which they are vector multiplied. The collective outcome (another vector) is subsequently passed to the next layer. The system as a whole generates the prediction, and can be used for generating highly complex predictions due to its [nonlinearity](https://www.machinecurve.com/index.php/2019/06/11/why-you-shouldnt-use-a-linear-activation-function/).

- [Read more about weights and features here.](https://www.machinecurve.com/index.php/2019/08/22/what-is-weight-initialization/)
- [Read here why full random weight initialization could not be a good idea.](https://www.machinecurve.com/index.php/2019/08/30/random-initialization-vanishing-and-exploding-gradients/)

### Why keep weights that don't contribute?

Now, you can possibly imagine that the contribution of each individual weight to model performance is not equal. Just like a group of people that attempt to reach a common goal, the input of some people is more important than the input of others. This could be unconscious - for example, because somebody is having a bad day - or on purposes. Whichever it is, it doesn't matter - just the absolute is what does.

Now, if some people (or in our case, neural network weights) do not contribute significantly, it could be that the cost of keeping them in (in terms of model sparsity and hence optimization) is larger than removing them from the model. That's precisely what **pruning** does: remove weights that do not contribute from your machine learning model. It does so quite ingeneously, as we shall see.

#### Saving storage and making things faster with magnitude-based pruning

In TensorFlow, we'll prune our models using **magnitude-based pruning**. This method, which is really simple, removes the smallest weight after each epoch (Universität Tubingen, n.d.). In fact, the pruning method is so simple that it compares the absolute size of the weight with some threshold lambda (Nervana Systems, n.d.):

\[latex\]thresh(w\_i)=\\left\\lbrace \\matrix{{{w\_i: \\; if \\;|w\_i| \\; \\gt}\\;\\lambda}\\cr {0: \\; if \\; |w\_i| \\leq \\lambda} } \\right\\rbrace\[/latex\]

According to Universität Tubingen (n.d.), this method often yields quite good results - no worse than more advanced methods.

Why this method works is because of the effect of weights that are set to zero. As we recall, within a neuron, some input vector \[latex\]\\textbf{x}\[/latex\] is multiplied with the weights vector \[latex\]\\textbf{w}\[/latex\]. If the weights in the vector are set to zero, the outcome will always be zero. This, in effect, ensures that the neuron no longer contributes to model performance.

Why, was the question I now had. Why does setting model weights to zero help optimize a model, and make it smaller? Gale et al. (2019) answer this question: "models can be stored and transmitted compactly using sparse matrix formats". This benefits from the fact that "\[sparse\] data is by nature more easily [compressed](https://en.wikipedia.org/wiki/Data_compression) and thus requires significantly less [storage](https://en.wikipedia.org/wiki/Computer_data_storage)." (Wikipedia, 2003). In addition, beyond compression, computation-wise programming code (such as computing `x`+`y`) can be made faster (e.g., it can be omitted if `x` or `y` are sparse, or both - `x+0` = `x`, and so on), benefiting processing - _inference_, in our case.

#### Now what happens to my accuracy?

Okay, fair enough - the simplicity of magnitude-based pruning combined with the benefits of sparse matrices definitely helps optimize your model. But what does this mean for model performance?

Often, not much. The weights that contribute to model performance most significantly often do not get removed. Still, this does mean that you observe _minor_ performance deterioration. For those cases, it is possible to fine-tune your model after pruning. This means that when pruning was performed (whether that means after an epoch or after you have finished training an early version of your model), it's possible to continue training; then, your model will attempt to get back to convergence with only a minority of the weights.

* * *

## Pruning: a Keras example

Great! We now know what pruning is all about, and most specifically, we understand how _magnitude-based pruning_ can benefit from storage and computational benefits related to sparse matrices. And who doesn't love its simplicity? :) That's why it's time to move from theory into practice, and see whether we can actually create a Keras model to which we apply pruning.

### Installing the TensorFlow Model Optimization toolkit

For pruning, we'll be using the TensorFlow Model Optimization toolkit, which "minimizes the complexity of optimizing machine learning inference." (TensorFlow Model Optimization, n.d.). It's a collection of interesting tools for optimizing your TensorFlow models.

You must first install it using `pip`, so that would be your first step to take:

```
pip install --user --upgrade tensorflow-model-optimization
```

### Using our Keras ConvNet

In another blog post, we saw how to create a [Convolutional Neural Network with Keras](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/). Here, I'll re-use that code, for its sheer simplicity - it does nothing more than create a small CNN and train it with the MNIST dataset. It'll be the starting point of our pruning exercise. Here it is - if you wish to understand it in more detail, I'd recommend taking a look at the page we just linked to before:

```
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tempfile
import tensorflow_model_optimization as tfmot
import numpy as np

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 10
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()
input_shape = (img_width, img_height, 1)

# Reshape data for ConvNet
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize [0, 255] into [0, 1]
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

Also make sure to store your model to a _temporary file_, so that you can compare the sizes of the original and the pruned model later:

```
# Store file 
_, keras_file = tempfile.mkstemp('.h5')
save_model(model, keras_file, include_optimizer=False)
print(f'Baseline model saved: {keras_file}')
```

### Loading and configuring pruning

Time to add pruning functionality to our model code!

We'll first add this:

```
# Load functionality for adding pruning wrappers
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
```

What it does is loading the `prune_low_magnitude` [functionality](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/prune_low_magnitude) from TensorFlow (Tfmot.sparsity.keras.prune\_low\_magnitude, n.d.). `prune_low_magnitude` simply modifies a layer by making it ready for pruning. It does so by wrapping a `keras` model with pruning functionality, more specifically by ensuring that the model's layers are prunable. This only _loads_ the functionality, we'll actually call it later.

Upon loading the pruning wrappers, we will set pruning configuration:

```
# Finish pruning after 5 epochs
pruning_epochs = 5
num_images = input_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs

# Define pruning configuration
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.40,
                                                               final_sparsity=0.70,
                                                               begin_step=0,
                                                               end_step=end_step)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
```

Here, the following happens:

- We configure the length of the pruning process by means of the number of `epochs` that the model will prune for, and fine-tune.
- We load the number of images used in our training set, minus the validation data.
- We compute the `end_step` of our pruning process given batch size, the number of images as well as the number of epochs.
- We subsequently define configuration for the pruning operation through `pruning_params`. We define a pruning schedule using `PolynomialDecay`, which means that sparsity of the model increases with increasing number of `epochs`. Initially, we set the model to be 40% sparse, increasingly getting sparser to eventually 70%. We begin at 0, and end at `end_step`.
- Finally, we actually call the `prune_low_magnitude` functionality (which generates the prunable model) from our initial `model` and the defined `pruning_params`.

**Suggestion:** make sure to read our [article about PolynomialDecay and ConstantSparsity pruning schedules](https://www.machinecurve.com/index.php/2020/09/29/tensorflow-pruning-schedules-constantsparsity-and-polynomialdecay/) to find out more about these particular schedules.

### Starting the pruning process

After configuring the pruning process, we can actually recompile the model (this is necessary because we added pruning functionality), and start the pruning process. We must use the `UpdatePruningStep` callback here, because it propagates optimizer activities to the pruning process (Tfmot.sparsity.keras.UpdatePruningStep, n.d.).

```
# Recompile the model
model_for_pruning.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Model callbacks
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

# Fitting data
model_for_pruning.fit(input_train, target_train,
                      batch_size=batch_size,
                      epochs=pruning_epochs,
                      verbose=verbosity,
                      callbacks=callbacks,
                      validation_split=validation_split)
```

### Measuring pruning effectiveness

Once pruning finishes, we must measure its effectiveness. We can do so in two ways:

- By measuring how much performance has changed, compared to before pruning;
- By measuring how much model size has changed, compared to before pruning.

We'll do so with the following lines of code:

```
# Generate generalization metrics
score_pruned = model_for_pruning.evaluate(input_test, target_test, verbose=0)
print(f'Pruned CNN - Test loss: {score_pruned[0]} / Test accuracy: {score_pruned[1]}')
print(f'Regular CNN - Test loss: {score[0]} / Test accuracy: {score[1]}')
```

Those ones are simple. They evaluate the pruned model with the testing data and subsequently print the outcome, as well as the (previously obtained) outcome of the original model.

Next, we export it again - just like we did before - to ensure that we can compare it:

```
# Export the model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
_, pruned_keras_file = tempfile.mkstemp('.h5')
save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print(f'Pruned model saved: {keras_file}')
```

Subsequently (thanks to Pruning Keras Example (n.d.)) we can compare the size of the Keras model. To illustrate the benefits of pruning, we must use a compression algorithm like `gzip`, after which we can compare the sizes of both models. Recall that pruning generates sparsity, and that sparse matrices can be saved very efficiently when compressed. That's why `gzip`s are useful for demonstration purposes. We first create a `def` that can be used for compression, and subsequently call it twice:

```
# Measuring the size of your pruned model
# (source: https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras#fine-tune_pre-trained_model_with_pruning)

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
```

### Runtime outcome

Now, it's time to run it. Save your file as e.g. `pruning.py`, and run it from a Python environment where you have `tensorflow` 2.x installed as well as `numpy` and the `tensorflow_model_optimization` toolkit.

First, regular training will start, followed by the pruning process, and then effectiveness is displayed on screen. First, with respect to model performance (i.e., loss and accuracy):

```
Pruned CNN - Test loss: 0.0218335362634185 / Test accuracy: 0.9923999905586243
Regular CNN - Test loss: 0.02442687187876436 / Test accuracy: 0.9915000200271606
```

The pruned model even performs slightly better than the regular one. This is likely because we trained the initial model for only 10 epochs, and subsequently continued with pruning afterwards. It's very much possible that the model had not yet converged; that moving towards convergence has continued in the pruning process. Often, performance deteriorates a bit, but should do so only slightly.

Then, with respect to model size:

```
Size of gzipped baseline Keras model: 1601609.00 bytes
Size of gzipped pruned Keras model: 679958.00 bytes
```

Pruning definitely made our model smaller - 2.35 times!

### Full model code

If you wish to obtain the full model code at once - here you go:

```
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tempfile
import tensorflow_model_optimization as tfmot
import numpy as np

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 10
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()
input_shape = (img_width, img_height, 1)

# Reshape data for ConvNet
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize [0, 255] into [0, 1]
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
print(f'Regular CNN - Test loss: {score[0]} / Test accuracy: {score[1]}')

# Store file 
_, keras_file = tempfile.mkstemp('.h5')
save_model(model, keras_file, include_optimizer=False)
print(f'Baseline model saved: {keras_file}')

# Load functionality for adding pruning wrappers
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Finish pruning after 5 epochs
pruning_epochs = 5
num_images = input_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs

# Define pruning configuration
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.40,
                                                               final_sparsity=0.70,
                                                               begin_step=0,
                                                               end_step=end_step)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Recompile the model
model_for_pruning.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Model callbacks
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

# Fitting data
model_for_pruning.fit(input_train, target_train,
                      batch_size=batch_size,
                      epochs=pruning_epochs,
                      verbose=verbosity,
                      callbacks=callbacks,
                      validation_split=validation_split)

# Generate generalization metrics
score_pruned = model_for_pruning.evaluate(input_test, target_test, verbose=0)
print(f'Pruned CNN - Test loss: {score_pruned[0]} / Test accuracy: {score_pruned[1]}')
print(f'Regular CNN - Test loss: {score[0]} / Test accuracy: {score[1]}')

# Export the model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
_, pruned_keras_file = tempfile.mkstemp('.h5')
save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print(f'Pruned model saved: {keras_file}')

# Measuring the size of your pruned model
# (source: https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras#fine-tune_pre-trained_model_with_pruning)

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
```

* * *

## Combining Pruning with Quantization for compound optimization

Above, we saw how we can apply **pruning** to our TensorFlow model to make it smaller without losing much performance. Doing so, we achieved a model that was 2.35 times smaller than the original one. However, it's possible to make the model even smaller. We can do so by means of [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/). If you're interested in what it is, I'd recommend you read the blog post for much detail. Here, we'll look at it very briefly and subsequently add it to our Keras example to gain even further improvements in model size.

### What is quantization?

Quantization, in short, means to change the number representation of your machine learning model (whether that's weights or also activations) in order to make it smaller.

By default, TensorFlow and Keras work with `float32` format. Using 32-bit floating point numbers, it's possible to store really large numbers with great precision. However, the fact that 32 bits can be used makes the model not so efficient in terms of storage - and neither in terms of speed (`float` operations are usually best run on GPUs, and this is cumbersome if you want to deploy your model in the field).

Quantization means changing this number representation. For example, using `float16` quantization, one can convert parts of the model from `float32` into `float16` format - approximately reducing model size by 50%, without losing much performance. Other approaches allow you to quantize into `int8` format (possibly losing quite some performance while gaining 4x size boost) or combined `int8`/`int16` format (best of both worlds). Fortunately, it's also possible to make your model quantization-aware, meaning that it simulates quantization during training so that the layers can already adapt to performance loss incurred by quantization.

In short, once the model has been pruned - i.e., stripped off non-contributing weights - we can subsequently add quantization. It should make the model even smaller in a compound way: 2.35 times size reduction should theoretically, using `int8` quantization, mean a 4 x 2.35 = 9.4 times reduction in size!

### Adding quantization to our Keras example

Let's now take a look how we can add quantization to a pruned TensorFlow model. More specifically, we'll add [dynamic range quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/#post-training-dynamic-range-quantization), which quantizes the weights, but not necessarily model activations.

Adding quantization first requires you to add a `TFLite` converter. This converter converts your TensorFlow model into TensorFlow Lite equivalent, which is what quantization will run against. Converting the model into a Lite model allows us to specify a model optimizer - `DEFAULT` or dynamic range quantization, in our case. Finally, we `convert()` the model:

```
# Convert into TFLite model and convert with DEFAULT (dynamic range) quantization
stripped_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
converter = tensorflow.lite.TFLiteConverter.from_keras_model(stripped_model)
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

Note that we must first strip the pruning wrappers from the model, creating a `stripped_model`. When the model has completed quantization, we can save it and print its size to see how much things have improved:

```
# Save quantized model
_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(tflite_model)
  
# Additional details
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))
```

Running again yields:

```
Size of gzipped baseline Keras model: 1601609.00 bytes
Size of gzipped pruned Keras model: 679958.00 bytes
Size of gzipped pruned and quantized TFlite model: 186745.00 bytes
```

...meaning:

- Size improvement original --> pruning: 2.35x
- Size improvement pruning --> quantization: 3.64x
- Total size improvement pruning + quantization: 8.58x

Almost 9 times smaller! 😎

### Full model code: pruning + quantization

Should you wish to run the pruning and quantization code at once, here you go:

```
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tempfile
import tensorflow_model_optimization as tfmot
import numpy as np

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 10
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()
input_shape = (img_width, img_height, 1)

# Reshape data for ConvNet
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize [0, 255] into [0, 1]
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
print(f'Regular CNN - Test loss: {score[0]} / Test accuracy: {score[1]}')

# Store file 
_, keras_file = tempfile.mkstemp('.h5')
save_model(model, keras_file, include_optimizer=False)
print(f'Baseline model saved: {keras_file}')

# Load functionality for adding pruning wrappers
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Finish pruning after 5 epochs
pruning_epochs = 5
num_images = input_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs

# Define pruning configuration
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.40,
                                                               final_sparsity=0.70,
                                                               begin_step=0,
                                                               end_step=end_step)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Recompile the model
model_for_pruning.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Model callbacks
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

# Fitting data
model_for_pruning.fit(input_train, target_train,
                      batch_size=batch_size,
                      epochs=pruning_epochs,
                      verbose=verbosity,
                      callbacks=callbacks,
                      validation_split=validation_split)

# Generate generalization metrics
score_pruned = model_for_pruning.evaluate(input_test, target_test, verbose=0)
print(f'Pruned CNN - Test loss: {score_pruned[0]} / Test accuracy: {score_pruned[1]}')
print(f'Regular CNN - Test loss: {score[0]} / Test accuracy: {score[1]}')

# Export the model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
_, pruned_keras_file = tempfile.mkstemp('.h5')
save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print(f'Pruned model saved: {keras_file}')

# Measuring the size of your pruned model
# (source: https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras#fine-tune_pre-trained_model_with_pruning)

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))

# Convert into TFLite model and convert with DEFAULT (dynamic range) quantization
stripped_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
converter = tensorflow.lite.TFLiteConverter.from_keras_model(stripped_model)
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save quantized model
_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(tflite_model)
  
# Additional details
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))
```

* * *

## Summary

This article demonstrated how TensorFlow models can be optimized using pruning. By means of pruning, which means to strip off weights that contribute insufficiently to model outcomes, models can be made sparser. Sparse models, in return, can be stored more efficiently, and can also _run_ more efficiently due to smart run-time effects in many programming languages and frameworks.

Beyond theory, we also looked at a practical scenario - where you're training a Convolutional Neural Network using Keras. After training, we first applied pruning using `PolynomialDecay`, which reduced model size 2.35 times. Then, we also added quantization - which we covered in another blog post but means changing the number representation of your model - and this reduced model size even further, to a total size reduction of 8.5 times compared to our initial model. Awesome!

I hope you have learnt a lot about model optimization from this blog article. I myself did when researching pruning and quantization! If you have any questions or remarks, please feel free to leave a comment in the comments section below 💬 I'm looking forward to hearing from you. Thank you for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

* * *

## References

Universität Tübingen. (n.d.). _Magnitude based pruning_. Kognitive Systeme | Universität Tübingen. [https://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node249.html](https://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node249.html)

_Trim insignificant weights_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/guide/pruning](https://www.tensorflow.org/model_optimization/guide/pruning)

_YOLOv5 is here_. (2020, August 4). Roboflow Blog. [https://blog.roboflow.com/yolov5-is-here](https://blog.roboflow.com/yolov5-is-here)

_Is faster RCNN the same thing as VGG-16, RESNET-50, etc... or not?_ (n.d.). Data Science Stack Exchange. [https://datascience.stackexchange.com/questions/54548/is-faster-rcnn-the-same-thing-as-vgg-16-resnet-50-etc-or-not](https://datascience.stackexchange.com/questions/54548/is-faster-rcnn-the-same-thing-as-vgg-16-resnet-50-etc-or-not)

_VGG16 - Convolutional network for classification and detection_. (2018, November 21). Neurohive - Neural Networks. [https://neurohive.io/en/popular-networks/vgg16/](https://neurohive.io/en/popular-networks/vgg16/)

Dwivedi, P. (2019, March 27). _Understanding and coding a ResNet in Keras_. Medium. [https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33)

Keras Team. (n.d.). _Keras documentation: Keras applications_. Keras: the Python deep learning API. [https://keras.io/api/applications/](https://keras.io/api/applications/)

_TensorFlow model optimization: An introduction to quantization – MachineCurve_. (2020, September 16). MachineCurve. [https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/)

_Decision tree pruning_. (2006, June 7). Wikipedia, the free encyclopedia. Retrieved September 22, 2020, from [https://en.wikipedia.org/wiki/Decision\_tree\_pruning](https://en.wikipedia.org/wiki/Decision_tree_pruning)

_Pruning - Neural network distiller_. (n.d.). Site not found · GitHub Pages. [https://nervanasystems.github.io/distiller/algo\_pruning.html](https://nervanasystems.github.io/distiller/algo_pruning.html)

Gale, T., Elsen, E., & Hooker, S. (2019). [The state of sparsity in deep neural networks](https://arxiv.org/pdf/1902.09574.pdf). _arXiv preprint arXiv:1902.09574_.

_Sparse matrix_. (2003, October 15). Wikipedia, the free encyclopedia. Retrieved September 22, 2020, from [https://en.wikipedia.org/wiki/Sparse\_matrix](https://en.wikipedia.org/wiki/Sparse_matrix)

_Computational advantages of sparse matrices - MATLAB & Simulink_. (n.d.). MathWorks - Makers of MATLAB and Simulink - MATLAB & Simulink. [https://www.mathworks.com/help/matlab/math/computational-advantages-of-sparse-matrices.html](https://www.mathworks.com/help/matlab/math/computational-advantages-of-sparse-matrices.html)

_TensorFlow model optimization_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/guide](https://www.tensorflow.org/model_optimization/guide)

_Tfmot.sparsity.keras.prune\_low\_magnitude_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/sparsity/keras/prune\_low\_magnitude](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/prune_low_magnitude)

_Tfmot.sparsity.keras.UpdatePruningStep_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/sparsity/keras/UpdatePruningStep](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/UpdatePruningStep)

_Pruning in Keras example_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/guide/pruning/pruning\_with\_keras#fine-tune\_pre-trained\_model\_with\_pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras#fine-tune_pre-trained_model_with_pruning)
