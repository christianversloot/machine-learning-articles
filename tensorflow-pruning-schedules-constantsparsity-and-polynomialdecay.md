---
title: "TensorFlow pruning schedules: ConstantSparsity and PolynomialDecay"
date: "2020-09-29"
categories: 
  - "frameworks"
tags: 
  - "constant-sparsity"
  - "edge-ai"
  - "optimizer"
  - "polynomial-decay"
  - "pruning"
  - "sparsity"
  - "tensorflow"
  - "model-optimization"
---

Today's deep learning models can become very large. That is, the weights of some contemporary model architectures are already approaching 500 gigabytes if you're working with pretrained models. In those cases, it is very difficult to run the models on embedded hardware, requiring cloud technology to run them successfully for model inference.

This is problematic when you want to generate predictions in the field that are accurate. Fortunately, today's deep learning frameworks provide a variety of techniques to help make models smaller and faster. In other blog articles, we covered two of those techniques: [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/) and [magnitude-based pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/). Especially when combining the two, it is possible to significantly reduce the size of your deep learning models for inference, while making them faster and while keeping them as accurate as possible.

They are interesting paths to making it possible to run your models at the edge, so I'd recommend the linked articles if you wish to read more. In this blog post, however, we'll take a more in-depth look at pruning in TensorFlow. More specifically, we'll first take a look at pruning by providing a brief and high-level recap. This allows the reader who hasn't read the posts linked before to get an idea what we're talking about. Subsequently, we'll be looking at the TensorFlow Model Optimization API, and specifically the `tfmot.sparsity.keras.PruningSchedule` functionality, which allows us to use preconfigured or custom-designed pruning schedules.

Once we understand `PruningSchedule`, it's time to take a look at two methods for pruning that come with the TensorFlow Model Optimization toolkit: the `ConstantSparsity` method and the `PolynomialDecay` method for pruning. We then converge towards a practical example with Keras by using `ConstantSparsity` to make our model sparser. If you want to get an example for `PolynomialDecay`, click [here](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/#pruning-a-keras-example) instead.

Enough introduction for now! Let's start :)

* * *

\[toc\]

* * *

## A brief recap on Pruning

If we train a machine learning model by means of a training, validation and testing dataset, we're following a methodology that is called [supervised learning](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process). If you look at the name, it already tells you much about how it works: by _supervising_ the learning process, you'll allow the model to learn generate successful predictions for new situations. Supervision, here, means to let the model learn and check its predictions with the true outcome later. It is a highly effective form of machine learning and is used very often in today's machine learning settings.

### Training a machine learning model: the iterative learning process

If we look at supervised learning in more detail, we can characterize it as follows:

![](images/High-level-training-process-1024x973.jpg)

We start our training process with a model where the weights are [initialized pseudorandomly](https://www.machinecurve.com/index.php/2019/08/30/random-initialization-vanishing-and-exploding-gradients/), with a small alteration given vanishing and exploding gradients. A model "weight" is effectively a vector that contains (part of the) learnt ability, and stores it numerically. All model weights, which are stored in a hierarchical fashion through layers, together capture all the patterns that have been learnt during training. Generating a new prediction involves a vector multiplication between the first-layer weight vectors and the vector of your input sample, subsequently passing the output to the next layer, and repeating the process for all downstream layers. The end result is one prediction, which can be a predicted class or a regressed real-valued number.

In terms of the machine learning process outlined above, we call feeding the training data to the model a _forward pass_. When data is passed forward, a prediction is computed for the input vector. In fact, this is done for all input vectors, generating as many predictions as there are training rows. Now that all the predictions are in, we can compare them with the ground truth - hence the supervision. In doing so, we can compute an average that represents the average error in the model, called a _[loss value](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/)_. Using this loss value, we can subsequently compute the error contribution of individual neurons and subsequently perform optimization using [gradient descent](https://www.machinecurve.com/index.php/2019/10/24/gradient-descent-and-its-variants/) or [modern optimizers](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/).

Repeating this process allows us to continuously adapt our weights until the loss value is lower than a predefined threshold, after which we (perhaps [automatically](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/)) stop the training process.

### Model optimization: pruning and quantization

Many of today's state-of-the-art machine learning architectures are [really big](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/#the-need-for-model-optimization) - 100 MB is no exception, and some architectures are 500 MB when they are trained. As we understand from the introduction and the linked article, it's highly impractical if not impossible to run those models adequately on embedded hardware, such as devices in the field.

They will then either be too _slow_ or they _cannot be loaded altogether_.

Using [pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/) and [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/), we can attempt to reduce model size. We studied pruning in detail in a different blog article. Let's now briefly cover what it is before we continue by studying the different types of pruning available in TensorFlow.

### Applying pruning to keep the important weights only

If we train a machine learning model, we can attempt to find out how much every model weight contributes to the final outcome. It should be clear that if a weight does not contribute significantly, it is not worth it to keep it in the model. In fact, there are many reasons why those weights should be thrown out - a.k.a., set to zero, making things _sparse_, as this is called:

- Compressing the model will be much more effective given the fact that sparse data can be compressed much better, decreasing the requirements for model storage.
- Running the model will be faster because sparse representations will always produce zero outputs (i.e., multiplying anything with 0 yields 0). Programmatically, this means that libraries don't have to perform vector multiplications when weights are sparse - making the prediction faster.
- Loading the model on embedded software will also be faster given the previous two reasons.

This is effectively what pruning does: it checks which weights contribute most, and throws out everything else that contributes less than a certain threshold. This is called [magnitude-based pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/) and is applied in TensorFlow. Since pruning happens during training, the weights that _do_ contribute significantly enough can adapt to the impact of the weights-thrown-out, making the model as a whole robust against sparsity on the fly.

While one must be very cautious still, since pruning (and quantization) can significantly impact model performance, both pruning and quantization can be great methods for optimizing your machine learning models.

* * *

## Pruning in TensorFlow

Now that we know how supervised machine learning models are trained and how pruning works conceptually, we can take a look at how TensorFlow provides methods for pruning. Specifically, this is provided through the TensorFlow Model Optimization toolkit, which must be installed separately (and is no core feature of TF itself, but integrates natively).

For pruning, it provides two methods:

- `ConstantSparsity` based pruning, which means that sparsity is kept constant during training.
- `PolynomialDecay` based pruning, which means that the degree of sparsity is changed during training.

### Generic terminology

Before we can look into `ConstantSparsity` and `PolynomialDecay` pruning schedules in more detail, we must take a look at some generic terminology first. More specifically, we'll discuss pruning schedules - implemented by means of a `PruningSchedule` - as well as pruning steps.

#### Pruning schedule

Applying pruning to a TensorFlow model must be done by means of a **pruning schedule** (PruningSchedule, n.d.). It "specifies when to prune layer and the sparsity(%) at each training step". More specifically:

> PruningSchedule controls pruning during training by notifying at each step whether the layer's weights should be pruned or not, and the sparsity(%) at which they should be pruned.

Essentially, it provides the necessary wrapper for pruning to take place in a scalable way. That is, while the pruning schedule _instance_ (such as `ConstantSparsity`) determines how pruning must be done, the `PruningSchedule` class provides the _skeleton_ for communicating the schedule. That is, it produces information about whether a layer should be pruned at a particular pruning step (by means of `should_prune`) and if so, what sparsity it must be pruned for.

#### Pruning steps

Now that we know about a `PruningSchedule`, we understand that it provides the skeleton for a pruning schedule to work. Any pruning schedule instance will thus tell you about whether pruning should be applied and what sparsity should be generated, but it will do so for a particular _step._ This terminology - **pruning steps** \-confused me, because well, what is a step? Is it equal to an epoch? If it is, why isn't it called epoch? If it's not, what is it?

In order to answer this question, I first looked at the source code for `PruningSchedule` on GitHub. As we know, TensorFlow is open source, and hence its code is available for everyone to see (TensorFlow/model-optimization, 2020). While it provides code that outputs whether to prune (`_should_prune_in_step`), it does not provide any explanation for the concept of a step.

However, in the [article about pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/), we saw that we must add the `UpdatePruningStep` callback to the part where pruning is applied. That is, after an epoch or a batch, it is applied to the model in question (Keras Team, n.d.). For this reason, it would be worthwhile to continue the search in the source code for the `UpdatePruningStep` callback.

Here, we see the following:

```
  def on_train_batch_begin(self, batch, logs=None):
    tuples = []
    for layer in self.prunable_layers:
      tuples.append((layer.pruning_step, self.step))

    K.batch_set_value(tuples)
    self.step = self.step + 1
```

This code is executed _upon the start of every batch_. To illustrate, if your training set has 1000 samples and you have a batch size of 250, every epoch will consist of 4 batches. Per epoch, the code above will be called 4 times.

In it, the pruning step is increased by one: `self.step = self.step + 1`.

This means that every _batch_ during your training process represents a pruning step. This is also why in the pruning article, [we configured the end\_step](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/#loading-and-configuring-pruning) as follows:

```
end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs
```

That's the number of images divided by the batch size (i.e., the number of steps per epoch) times the number of epochs; this produces the total number of steps performed during pruning.

### ConstantSparsity based pruning

TensorFlow's **constant sparsity** during pruning can be characterized as follows (ConstantSparsity, n.d.):

> Pruning schedule with constant sparsity(%) throughout training.

As it inherits from the `PruningSchedule` defined above, it must implement all the Python definitions and can hence be used directly in pruning.

It accepts the following arguments (source: [TensorFlow](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/ConstantSparsity) - [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), no edits):

| Args |
| --- |
| `target_sparsity` | A scalar float representing the target sparsity value. |
| `begin_step` | Step at which to begin pruning. |
| `end_step` | Step at which to end pruning. `-1` by default. `-1` implies continuing to prune till the end of training. |
| `frequency` | Only apply pruning every `frequency` steps. |

Those arguments allow you to configure pruning to your needs. With a constant target sparsity, you set the degree of sparsity that is to be applied when sparsity must be applied. This latter is determined by the `begin_step` as well as the `end_step` and `frequency`. Should you wish to apply pruning only to the final part of training, you can configure so through `begin_step`; the same goes for applying to the entire training process, only the first part, and other configurations. How often must be pruned can be configured by means of `frequency` (default frequency = 100).

It returns the following (TensorFlow/model-optimization, n.d.) data with respect to `should_prune` and `sparsity`:

```
    return (self._should_prune_in_step(step, self.begin_step, self.end_step,
                                       self.frequency),
            tf.constant(self.target_sparsity, dtype=tf.float32))
```

It thus indeed returns a constant sparsity value to prune for.

### PolynomialDecay based pruning

If you recall some basic maths, you might remember what is known as a _polynomial function_. Such functions, for example `x` squared, take an input `x` and multiply it by some value of themselves. This can also be applied in pruning to make the applied pruning level non-constant. Using **polynomial decay based sparsity**, more or fewer sparsity can be used with increasing or decreasing speed, as training progresses. It is represented in the `PolynomialDecay` function with TensorFlow:

> Pruning Schedule with a PolynomialDecay function.

It also inherits from `PruningSchedule`, so it implements all the necessary functionality for it to be used in pruning directly.

It accepts the following arguments (source: [TensorFlow](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/ConstantSparsity) - [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), no edits):

| Args |
| --- |
| `initial_sparsity` | Sparsity (%) at which pruning begins. |
| `final_sparsity` | Sparsity (%) at which pruning ends. |
| `begin_step` | Step at which to begin pruning. |
| `end_step` | Step at which to end pruning. |
| `power` | Exponent to be used in the sparsity function. |
| `frequency` | Only apply pruning every `frequency` steps. |

Here, the user must provide an _initial sparsity_ as well as a _final sparsity_ percentage. Similar to constant sparsity, a begin and end step and a frequency must be passed along as well. New again is the `power` argument, which represents the exponent of the polynomial function to be used for computing sparsity.

### When use which pruning schedule?

It seems that there has been no extensive investigation into what pruning schedule must be used under what conditions. For example, Zhu and Gupta (2017) have investigated the effects of what we know as `PolynomialDecay` for a variety of sparsity levels ranging between 50 and 90%, and found that sparsity does not significantly hamper accuracy.

In my point of view - and I will test my point of view later in this blog - I think this partially occurs because how polynomial decay is implemented. Their training process started at a low sparsity level (0%, in fact) and was then increased (to 50/75/87.5% for three scenarios, respectively). In all scenarios, sparsity increase started at the 2000th step, in order to allow the model to start its path towards convergence without being hurt by sparsity inducing methods already.

The effect of this strategy is that while the model already starts converging, sparsity is introduced slowly. Model weights can take into account this impact and become robust to the effect of weights being dropped, similar to [quantization-aware training](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/#quantization-aware-training) in the case of quantization. I personally think that this is a better strategy compared to a `ConstantSparsity`, which immediately increases sparsity levels from 0% to the constant sparsity level that was configured.

* * *

## A code example with ConstantSparsity

Next, we will provide an example that trains a model with `ConstantSparsity` applied. The model code is equal to the version applying `PolynomialDecay`, but then applies `ConstantSparsity` instead for a sparsity of 87.5%. We start applying sparsity after 20% of the training process has finished, i.e. after `0.2 * end_step`, and continue pruning until `end_step` i.e. for the rest of the pruning steps.

We train for 30 epochs, as a ConvNet-based MNIST classifier will always see good performance after only few epochs.

Should you wish to get additional explanation or see the code for `PolynomialDecay`, click [here](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/#pruning-a-keras-example). Here is the full code for creating, training, pruning, saving and comparing a pruned Keras model with `ConstantSparsity`:

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
no_classes = 10
validation_split = 0.2
verbosity = 1
pruning_epochs = 30

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

# Load functionality for adding pruning wrappers
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Finish pruning after 10 epochs
num_images = input_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs

# Define pruning configuration
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.875,
                                                               begin_step=0.2*end_step,
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

# Export the model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
_, pruned_keras_file = tempfile.mkstemp('.h5')
save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print(f'Pruned model saved: {pruned_keras_file}')

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

print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
```

* * *

## Comparing the effects of ConstantSparsity and PolynomialDecay based pruning

Earlier, we saw that there has been no large-scale investigation into what method of pruning works best in TensorFlow. Although intuitively, it feels as if `PolynomialDecay` pruned models produce more robust models, this is simply an intuition and must be tested. Training the `ConstantSparsity` model with the classifier above for 30 epochs yields the following results:

```
Pruned CNN - Test loss: 0.03168991032061167 / Test accuracy: 0.9886000156402588
Size of gzipped pruned Keras model: 388071.00 bytes
```

It performs really well, but this is expected from models classifying MNIST digits.

Subsequently retraining the PolynomialDecay based one from the [other post](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/#pruning-a-keras-example), but then following the Zhu & Gupta (2017) setting (sparsity at 0% at first, up to 87.5% - equaling the constant sparsity of the other model; beginning at 20% of the training data), this is the outcome of training with polynomial decay:

```
Pruned CNN - Test loss: 0.02177981694244372 / Test accuracy: 0.9926999807357788
Size of gzipped pruned Keras model: 384305.00 bytes
```

Recall that the size of the baseline model trained in that other post was much larger:

```
Size of gzipped baseline Keras model: 1601609.00 bytes
```

In short, pruning seems to work both ways in terms of reducing model size. `PolynomialDecay` based sparsity seems to work slightly better (slightly higher accuracy and especially a 33% lower loss value). It also produced a smaller model in size. Now, while this is a N = 1 experiment, which cannot definitively answer whether it is better than `ConstantSparsity`, the intuitions are still standing. We challenge others to perform additional experiments in order to find out.

* * *

## Summary

In this article, we studied pruning in TensorFlow in more detail. Before, we covered quantization and pruning for model optimization, but for the latter there are multiple ways of doing so in TensorFlow. This blog post looked at those methods and their difference.

Before being able to compare the pruning schedules, we provided a brief recap to how supervised machine learning models are trained, and how they can be pruned. By discussing the forward pass, computation of the loss value and subsequently backward computation of the error and optimization, we saw how models are trained. We also saw what pruning does to weights, and how the sparsity this brongs benefits model storage, model loading and model inference, especially hardware at the edge.

Then, we looked at the pruning schedules available in TensorFlow: `ConstantSparsity` and `PolynomialDecay`. Both inheriting the `PruningSchedule` class, they provide functionalities that determine whether a particular layer must be pruned during a particular step, and to what sparsity. Generally, the constant sparsity applies a constant sparsity when it prunes a layer, while the polynomial decay pruning schedule induces a sparsity level based on a polynomial function, from a particular sparsity level to another.

Finally, we provided an example using Keras, TensorFlow's way of creating machine learning models. In comparing the outcomes, we saw that `PolynomialDecay` based sparsity / pruning works slightly better than `ConstantSparsity`, which was expected intuitively.

I hope you've learnt a lot by reading this post! I did, when researching :) Please feel free to leave a comment in the comments section below if you have any questions, remarks or other suggestions for improvement 💬 Thank you for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

* * *

## References

_Module: Tfmot.sparsity_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/sparsity](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity)

_Tfmot.sparsity.keras.ConstantSparsity_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/sparsity/keras/ConstantSparsity](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/ConstantSparsity)

_Tfmot.sparsity.keras.PolynomialDecay_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/sparsity/keras/PolynomialDecay](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PolynomialDecay)

_Tfmot.sparsity.keras.PruningSchedule_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/sparsity/keras/PruningSchedule](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PruningSchedule)

_TensorFlow/model-optimization_. (2020, 30). GitHub. [https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow\_model\_optimization/python/core/sparsity/keras/pruning\_callbacks.py#L46](https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow_model_optimization/python/core/sparsity/keras/pruning_callbacks.py#L46)

_TensorFlow/model-optimization_. (2020, January 10). GitHub. [https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow\_model\_optimization/python/core/sparsity/keras/pruning\_schedule.py#L41](https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow_model_optimization/python/core/sparsity/keras/pruning_schedule.py#L41)

_Tfmot.sparsity.keras.UpdatePruningStep_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/sparsity/keras/UpdatePruningStep](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/UpdatePruningStep)

Keras Team. (n.d.). _Keras documentation: Callbacks API_. Keras: the Python deep learning API. [https://keras.io/api/callbacks/](https://keras.io/api/callbacks/)

_TensorFlow/model-optimization_. (n.d.). GitHub. [https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow\_model\_optimization/python/core/sparsity/keras/pruning\_schedule.py#L137-L180](https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow_model_optimization/python/core/sparsity/keras/pruning_schedule.py#L137-L180)

_TensorFlow/model-optimization_. (n.d.). GitHub. [https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow\_model\_optimization/python/core/sparsity/keras/pruning\_schedule.py#L183-L262](https://github.com/tensorflow/model-optimization/blob/0f6dd5aeb818c5f61123fc1d5642435ea0f5cd70/tensorflow_model_optimization/python/core/sparsity/keras/pruning_schedule.py#L183-L262)

Zhu, M., & Gupta, S. (2017). [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878). _arXiv preprint arXiv:1710.01878_.
