---
title: "An introduction to TensorFlow.Keras callbacks"
date: "2020-11-10"
categories:
  - "frameworks"
tags:
  - "callbacks"
  - "keras"
  - "tensorflow"
---

Training a deep learning model is both simple and complex at the same time. It's simple because with libraries like TensorFlow 2.0 (`tensorflow.keras`, specifically) it's very easy to get started. But while creating a first model is easy, fine-tuning it while knowing what you are doing is a bit more complex.

For example, you will need some knowledge on the [supervised learning process](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process), gradient descent or other optimization, regularization, and a lot of other contributing factors.

Tweaking and tuning a deep learning models therefore benefits from two things: insight into what is happening and automated control to avoid the need for human intervention where possible. In Keras, this can be achieved with the `tensorflow.keras.callbacks` API. In this article, we will look into Callbacks in more detail. We will first illustrate what they are by displaying where they play a role in the supervised machine learning process. Then, we cover the Callbacks API - and for each callback, illustrate what it can be used for together with a small example. Finally, we will show how you can create your own Callback with the `tensorflow.keras.callbacks.Base` class.

Let's take a look :)

**Update 11/Jan/2021:** changed header image.

* * *

\[toc\]

* * *

## Callbacks and their role in the training process

In our article about the [supervised machine learning process](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process), we saw how a supervised machine learning model is trained:

1. A machine learning model (today, often a neural network) is [initialized](https://www.machinecurve.com/index.php/2019/08/22/what-is-weight-initialization/).
2. Samples from the training set are fed forward, through the model, resulting in a set of predictions.
3. The predictions are compared with what is known as the _ground truth_ (i.e. the labels corresponding to the training samples), resulting in one value - a [loss value](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions) - telling us how _bad_ the model performs.
4. Based on the loss value and the subsequent backwards computation of the error, the weights are changed a little bit, to make the model a bit better. Then, we're either moving back to step 2, or we stop the training process.

As we can see, steps 2-4 are _iterative_, meaning that the model improves in a cyclical fashion. This is reflected in the figure below.

![](images/High-level-training-process-1024x973.jpg)

In Machine Learning terms, each iteration is also called an **epoch**. Hence, training a machine learning model involves the completion of at least one, but often multiple epochs. Note from the article about [gradient descent based optimization](https://www.machinecurve.com/index.php/2019/10/24/gradient-descent-and-its-variants/) that we often don't feed forward all data at once. Instead, we use what is called a _minibatch approach_ - the entire batch of data is fed forward in smaller batches called minibatches. By consequence, each epoch consists of at least one but often multiple **batches** of data.

Now, it can be the case that you want to get insights from the training process while it is running. Or you want to provide automated steering in order to avoid wasting resources. In those cases, you might want to add a **callback** to your Keras model.

> A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc).
>
> Keras Team (n.d.)

As we shall see later in this article, among others, there are [callbacks for monitoring](https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/) and for stopping the training process [when it no longer makes the model better](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/). This is possible because with callbacks, we can 'capture' the training process while it is happening. They essentially 'hook' into the training process by allowing the training process to invoke certain callback definitions. In Keras, each callback implements at least one, but possibly multiple of the following definitions (Keras Team, n.d.).

- With the `on_train_begin` and `on_train_end` definitions, we can perform a certain action either when `model.fit` starts executing or when the training process has just ended.
- With the `on_epoch_begin` and `on_epoch_end` definitions, we can perform a certain action just before the start of an epoch, or directly after it has ended.
- With the `on_test_begin` and `on_test_end` definitions, we can perform a certain action just before or after the model [is evaluated](https://www.machinecurve.com/index.php/2020/11/03/how-to-evaluate-a-keras-model-with-model-evaluate/).
- With the `on_predict_begin` and `on_predict_end` definitions, we can do the same, but then when we generate [new predictions](https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/). If we predict for a batch rather than a single sample, we can use the `on_predict_batch_begin` and `on_predict_batch_end` definitions.
- With the `on_train_batch_begin`, `on_train_batch_end`, `on_test_batch_begin` and `on_test_batch_end` definitions, we can perform a certain action directly before or after we feed a batch to either the training or testing process.

As we can see, by using a callback, through the definitions outlined above, we can control the training process at a variety of levels.

* * *

## The Keras Callbacks API

Now that we understand what callbacks are, how they can help us, and what definitions - and hence hooks - are available for 'breaking into' your training process in TensorFlow 2.x based Keras. Now, it's time to take a look at the Keras Callbacks API. Available as `tensorflow.keras.callbacks`, it's a set of generally valuable Callbacks that can be used in a variety of cases.

Most specifically, it contains the following callbacks, and we will cover each of them next:

1. **ModelCheckpoint callback:** can be used to [automatically save a model](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/) after each epoch, or just the best one.
2. **TensorBoard callback:** allows us to monitor the training process in realtime with [TensorBoard](https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/).
3. **EarlyStopping callback:** ensures that the training process stops if the loss value [does no longer improve](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/).
4. **LearningRateScheduler callback:** updates the learning rate before the start of an epoch, based on a `scheduler` function.
5. **ReduceLROnPlateau callback:** reduces learning rate if the loss value does no longer improve.
6. **RemoteMonitor callback:** sends TensorFlow training events to a remote monitor, such as a logging system.
7. **LambdaCallback:** allows us to define simple functions that can be executed as a callback.
8. **TerminateOnNaN callback:** if the loss value is Not a Number (NaN), the training process stops.
9. **CSVLogger callback:** streams the outcome of an epoch to a CSV file.
10. **ProgbarLogger callback:** used to determine what is printed to standard output in the Keras progress bar.

### How do we add a callback to a Keras model?

Before we take a look at all the individual callbacks, we must take a look at how we can use the `tensorflow.keras.callbacks` API in the first place. Doing so is really simple and only changes your code in a minor way:

1. You must add the specific callbacks to the model imports.
2. You must _initialize_ the callbacks you want to use, including their configuration; preferably do so in a list.
3. You must add the callbacks to the `model.fit` call.

With those three simple steps, you ensure that the callbacks are hooked into the training process!

For example, if we want to use both `ModelCheckpoint` and `EarlyStopping` - [as we do here](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/) - for step (1), we first **add the imports**:

```python

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

Then, for step (2), we **initialize the callbacks** in a list:

```python
keras_callbacks = [
      EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.01),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
]
```

And then, for step (3), we simply **add the callbacks** to `model.fit`:

```python
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### ModelCheckpoint callback

If you want to periodically save your Keras model - or the model weights - to some file, the `ModelCheckpoint` callback is what you need.

> Callback to save the Keras model or model weights at some frequency.
>
> TensorFlow (n.d.)

It is available as follows:

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs
)
```

With the following arguments:

- With `filepath`, you can specify where the model must be saved.
- If you want to save only if some quantity has changed, you can set this quantity by means of `monitor`. It is set to validation loss by default.
- With `verbose`, you can specify if the callback output should be output in your standard output (often, your terminal).
- If you only want to save the model when the monitored quantity improves, you can set `save_best_only` to `True`.
- Normally, the entire model is [saved](https://www.machinecurve.com/index.php/2020/02/14/how-to-save-and-load-a-model-with-keras/) - that is, the stack of layers as well as the [model weights](https://www.machinecurve.com/index.php/2019/08/22/what-is-weight-initialization/). If you want to save the weights only (e.g. because you can initialize the model yourself), you can set `save_weights_only` to `True`.
- With `mode`, you can determine in what direction the `monitor` quantity must move to consider it to be an improvement. You can choose any from `{auto, min, max}`. When it is set to `auto`, it determines the `mode` based on the `monitor` - with loss, for example, it will be `min`; with accuracy, it will be `max`.
- The `save_freq` allows you to determine when to save the model. By default, it is saved after every epoch (or checks whether it has improved after every epoch). By changing the `'epoch'` string into an integer, you can also instruct Keras to save after every `n` minibatches.
- If you want, you can specify other compatible `options` as well. Check the `ModelCheckpoint` docs (see link in references) for more information about these `options`.

Using `ModelCheckpoint` is easy - and here is an example based on a [generator](https://www.machinecurve.com/index.php/2020/04/06/using-simple-generators-to-flow-data-from-file-with-keras/):

```python
checkpoint_path=f'{os.path.dirname(os.path.realpath(__file__))}/covid-convnet.h5'
keras_callbacks = [
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### TensorBoard callback

Did you know that you can visualize the training process realtime [with TensorBoard](https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/)?

![](images/image-1.png)

With the `TensorBoard` callback, you can link TensorBoard with your Keras model.

> Enable visualizations for TensorBoard.
>
> TensorFlow (n.d.)

The callback logs a range of items from the training process into your TensorBoard log location:

- Metrics summary plots
- Training graph visualization
- Activation histograms
- Sampled profiling

It is implemented as follows:

```python
tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None, **kwargs
)
```

- With `log_dir`, you can specify the file path to your TensorBoard log folder.
- The `TensorBoard` callback computes activation and weight histograms. With `histogram_freq`, you can specify the frequency (in epochs) when this should happen. Histograms will not be computed when `histogram_freq` is set to 0.
- Whether to write the TensorFlow graph to the logs can be configured with `write_graph`.
- If you want to visualize your model weights as images in TensorBoard, you can set `write_images` to `True`.
- With `update_freq`, you can specify when this callback sends data to TensorBoard. If it's set to `epoch`, it will send data every epoch. If set to `batch`, data will be sent on every batch. If set to an integer `n` instead, data will be sent every `n` batches.
- With the [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler), we can calculate the compute performance of TensorFlow - that is, the resources it needs at a point in time. With `profile_batch`, you can specify a batch to profile, meaning that Profiling information will be sent to TensorBoard as well.
- If you are using [Embeddings](https://www.machinecurve.com/index.php/2020/03/03/classifying-imdb-sentiment-with-keras-and-embeddings-dropout-conv1d/), it is possible to let TensorFlow visualize them. Specifying the `embeddings_freq` allows you to configure when Embeddings need to be visualized; it represents the frequency in epochs. Embeddings will not be visualized when the frequency is set to 0.
- A dictionary with Embeddings metadata can be passed along with `embeddings_metadata`.

Here is an example of using the `TensorBoard` callback within your Keras model:

```python
keras_callbacks = [
      TensorBoard(log_dir="./logs")
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### EarlyStopping callback

Optimizing your neural network involves applying [gradient descent](https://www.machinecurve.com/index.php/2019/10/24/gradient-descent-and-its-variants/) or [another optimizer](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/) to a loss value generated by feeding forward batches of training samples, generating predictions that are compared with the corresponding training labels.

During this process, you want to find a model that performs well in terms of predictions (i.e., it is not underfit) but that is not too rigid with respect to the dataset it is trained on (i.e., it is neither overfit). That's why the `EarlyStopping` callback can be useful if you are dealing with a situation like this.

> Stop training when a monitored metric has stopped improving.
>
> TensorBoard (n.d.)

It is implemented as follows:

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)
```

- The `monitor` is the quantity to monitor for improvement; it is similar to the quantity monitored for `ModelCheckpointing`.
- The same goes for the `mode`.
- With `min_delta`, you can configure the minimum change that must happen from the current `monitor` in order to consider the change an improvement.
- With `patience`, you can indicate how long in epochs to wait for additional improvements before stopping the training process.
- With `verbose`, you can specify the verbosity of the callback, i.e. whether the output is written to standard output.
- The `baseline` value can be configured to specify a minimum `monitor` that must be achieved at all before _any_ change can be considered an improvement.
- As you would expect, having a `patience` > 0 will ensure that the model is trained for `patience` more epochs, possibly making it worse. With `restore_best_weights`, we can restore the weights of the best-performing model instance when the training process stops. This can be useful if you directly perform [model evaluation](https://www.machinecurve.com/index.php/2020/11/03/how-to-evaluate-a-keras-model-with-model-evaluate/) after stopping the training process.

Here is an example of using `EarlyStopping` with Keras:

```python

keras_callbacks = [
      EarlyStopping(monitor='val_loss', min_delta=0.001, restore_best_weights=True)
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### LearningRateScheduler callback

During the optimization process, a so called _weight update_ is computed. However, if we compare the optimization process with rolling a ball down a mountain (reflecting the [loss landscape](https://www.machinecurve.com/index.php/2020/02/26/getting-out-of-loss-plateaus-by-adjusting-learning-rates/)), we want to smooth the ride, ensuring that our ball does not bounce out of control. That is why a [learning rate](https://www.machinecurve.com/index.php/2019/11/06/what-is-a-learning-rate-in-a-neural-network/) is applied: it specifies a fraction of the weight update to be used by the optimizer.

Preferably being relatively large during the early iterations and lower in the later stages, we must adapt the learning rate during the training process. This is called [learning rate decay](https://www.machinecurve.com/index.php/2019/11/11/problems-with-fixed-and-decaying-learning-rates/) and shows what a _learning rate scheduler_ can be useful for. The `LearningRateScheduler` callback implements this functionality.

> At the beginning of every epoch, this callback gets the updated learning rate value from `schedule` function provided at `__init__`, with the current epoch and current learning rate, and applies the updated learning rate on the optimizer.
>
> TensorFlow (n.d.)

Its implementation is really simple:

```python
tf.keras.callbacks.LearningRateScheduler(
    schedule, verbose=0
)
```

- It accepts a `schedule` function which you can use to decide yourself how the learning rate must be scheduled during every epoch.
- With `verbose`, you can decide to illustrate the callback output in your standard output.

Here is an example of using the `LearningRateScheduler` with Keras:

```python
def scheduler(epoch, learning_rate):
  if epoch < 15:
    return learning_rate
  else:
    return learning_rate * 0.99

keras_callbacks = [
      LearningRateScheduler(scheduler)
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### ReduceLROnPlateau callback

During the optimization process - i.e., rolling the ball downhill - it can be the case that you encounter so-called _loss plateaus_. In those areas, the gradient of the loss function is close to zero, but not entirely - indicating that you are in the vicinity of a loss minimum. That is, close to where you want to be (unless you are dealing with a local minimum, of course).

Keeping your learning rate equal when close to a plateau means that your model will likely not improve any further. This happens because your model will optimize, oscillating around the loss minimum, simply because the steps the current [learning rate](https://www.machinecurve.com/index.php/2019/11/06/what-is-a-learning-rate-in-a-neural-network/) it instructs to set are too big.

With the `ReduceLROnPlateau` callback, the optimization process can be instructed to _reduce_ the learning rate (and hence the step) when a plateau is encountered.

> Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
>
> TensorFlow (n.d.)

The callback is implemented as follows:

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0, **kwargs
)
```

- The `monitor` and `patience` resemble the monitors and patience values that we have already encountered. In other words, it is the quantity to observe that helps us judge whether improvement has happened. Patience tells us how long to wait before we consider improvement impossible. The `mode` is related to the `monitor` and instructs what kind of operation to perform while monitoring: `min` or `max` (or `auto`matically determined).
- The `min_delta` tells us _how much_ the model should improve at minimum before we consider the change an improvement.
- The `factor` determines how much to decrease the learning rate upon encountering a plateau: `new_lr = lr * factor`.
- The `verbose` attribute can be configured to display the callback output in your standard output.
- The `min_lr` gives us a lower bound on the learning rate.
- The `cooldown` attribute instructs the model to wait with invoking this specific callback for a number of epochs, allowing us to find _some improvement_ with the reduced learning rate (this could take a few epochs).

An example of using the `ReduceLROnPlateau` callback with Keras:

```python
keras_callbacks = [
      ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, cooldown=5, min_lr=0.000000001)
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### RemoteMonitor callback

Above, we saw that training logs can be distributed to [TensorBoard](https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/) for visualization and logging purposes. However, it can be the case that you have your own logging and visualization system - whether that's a cloud-based system or a locally installed Grafana or Elastic Stack visualization tooling.

In those cases, you might wish to send the training logs there instead. The `RemoteMonitor` callback can help you do this.

> Callback used to stream events to a server.
>
> TensorFlow (n.d.)

It is implemented as follows:

```python
tf.keras.callbacks.RemoteMonitor(
    root='http://localhost:9000', path='/publish/epoch/end/', field='data',
    headers=None, send_as_json=False
)
```

- With the `root` argument, you can specify the root of the endpoint to where data must be sent.
- The `path` indicates the path relative to `root` where data must be sent. In other words, `root + path` describe the full endpoint.
- The JSON field under which data is sent can be configured with `field`.
- In `headers`, additional HTTP headers (such as an Authorization header) can be provided.
- With `send_as_json` as `True`, the content type of the request will be changed to `application/json`. Otherwise, it will be sent as part of a form.

An example of using the `RemoteMonitor` callback with Keras:

```python
keras_callbacks = [
      RemoteMonitor(root='https://some-domain.com', path='/statistics/keras')
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### LambdaCallback

Say that you want a certain function to fire after every batch or every epoch - a simple function, nothing special. However, it's not provided in the collection of callbacks presented with the `tensorflow.keras.callbacks` API. In this case, you might want to use the `LambdaCallback`.

> Callback for creating simple, custom callbacks on-the-fly. This callback is constructed with anonymous functions that will be called at the appropriate time. Te
>
> TensorFlow (n.d.)

It can thus be used to provide anonymous (i.e. `lambda` functions without a name) functions to the training process. The callback looks as follows:

```python
tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None,
    on_train_begin=None, on_train_end=None, **kwargs
)
```

Here, the `on_epoch_begin`, `on_epoch_end`, `on_batch_begin`, `on_batch_end`, `on_train_begin` and `on_train_end` _event_ based arguments can be filled with Python definitions. They are executed at the right point in time.

An example of a `LambdaCallback` added to your Keras model:

```python
keras_callbacks = [
      LambdaCallback(on_batch_end=lambda batch, log_data: print(batch))
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### TerminateOnNaN callback

In some cases (e.g. when you did not apply min-max normalization to your input data), the loss value can be very strange - outputting values close to Infinity or values that are Not a Number (`NaN`). In those cases, you don't want to pursue further training. The `TerminateOnNaN` callback can help here.

> Callback that terminates training when a NaN loss is encountered.
>
> TensorFlow (n.d.)

It is implemented as follows:

```python
tf.keras.callbacks.TerminateOnNaN()
```

An example of using the `TerminateOnNaN` callback with your Keras model:

```python
keras_callbacks = [
      TerminateOnNaN()
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### CSVLogger callback

CSV files can be very useful when you need to exchange data. If you want to flush your training logs into a CSV file, the `CSVLogger` callback can be useful to you.

> Callback that streams epoch results to a CSV file.
>
> TensorFlow (n.d.)

It is implemented as follows:

```python
tf.keras.callbacks.CSVLogger(
    filename, separator=',', append=False
)
```

- The `filename` attribute determines where the CSV file is located. If there is none, it will be created.
- The `separator` attribute determines what character separates the columns in a single row, and is also called delimiter.
- With `append`, you can indicate whether data should simply be added to the end of the file, or a new file should overwrite the old one every time.

This is an example of using the `CSVLogger` callback with Keras:

```python
keras_callbacks = [
      CSVLogger('./logs.csv', separator=';', append=True)
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### ProgbarLogger callback

When you are training a Keras model with verbosity set to `True`, you will see a progress bar in your terminal. With the `ProgbarLogger` callback, you can change what is displayed there.

> Callback that prints metrics to stdout.
>
> TensorFlow (n.d.)

It is implemented as follows:

```python
tf.keras.callbacks.ProgbarLogger(
    count_mode='samples', stateful_metrics=None
)
```

- With `count_mode`, you can instruct Keras to display samples or steps (i.e. batches) already fed forward through the model
- The `stateful_metrics` attribute can contain metrics that should not be averaged over time.

Here is an example of using the `ProgbarLogger` callback with Keras.

```python
keras_callbacks = [
      ProgbarLogger(count_mode='samples')
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### Experimental: BackupAndRestore callback

When you are training a neural network, especially in a [distributed setting](https://www.machinecurve.com/index.php/2020/10/16/tensorflow-cloud-easy-cloud-based-training-of-your-keras-model/), it would be problematic if your training process suddenly stops - e.g. due to machine failure. Every iteration passed so far will be gone. With the experimental `BackupAndRestore` callback, you can instruct Keras to create temporary checkpoint files after each epoch, to which you can restore later.

> `BackupAndRestore` callback is intended to recover from interruptions that happened in the middle of a model.fit execution by backing up the training states in a temporary checkpoint file (based on TF CheckpointManager) at the end of each epoch.
>
> TensorFlow (n.d.)

It is implemented as follows:

```python
tf.keras.callbacks.experimental.BackupAndRestore(
    backup_dir
)
```

Here, the `backup_dir` attribute indicates the folder where checkpoints should be created.

Here is an example of using the `BackupAndRestore` callback with Keras.

```python
keras_callbacks = [
       BackupAndRestore('./checkpoints')
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

### Applied by default: History and BaseLogger callbacks

There are two callbacks that are part of the `tensorflow.keras.callbacks` API but which can be covered less extensively - because of the simple reason that they are already applied to each Keras model under the hood.

They are the `History` and `BaseLogger` callbacks.

- The `History` callback generates a `History` [object](https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training-process-in-keras/#the-history-object) when calling `model.fit`.
- The `BaseLogger` callback accumulates basic metrics to display later.

* * *

## Creating your own callback with the Base Callback

Sometimes, neither the default or the `lambda` callbacks can provide the functionality you need. In those cases, you can create your own callback, by using the Base callback class `tensorflow.keras.callbacks.Callback`. Creating one is very simple: you define a `class`, create the relevant definitions (you can choose from `on_epoch_begin`, `on_epoch_end`, `on_batch_begin`, `on_batch_end`, `on_train_begin` and `on_train_end` etc.), and then add the callback to your callbacks list. There you go!

```python
class OwnCallback(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print('Training is now beginning!')

keras_callbacks = [
       OwnCallback()
]
model.fit(train_generator,
          epochs=50,
          verbose=1,
          callbacks=keras_callbacks,
          validation_data=val_generator)
```

* * *

## Summary

In this article, we looked at the concept of a callback for hooking into the supervised machine learning training process. Sometimes, you want to receive _additional information_ while you are training a model. In other cases, you want to _actively steer the process_ into a desired direction. Both cases are possible by means of a callback.

Beyond the conceptual introduction to callbacks, we also looked at how Keras implements them - by means of the `tensorflow.keras.callbacks` API. We briefly looked at each individual callback provided by Keras, ranging from automated changes to hyperparameters to logging in TensorBoard, file or into a remote monitor. We also looked at creating your own callback, whether that's with a `LambdaCallback` for simple custom callbacks or with the Base callback class for more complex ones.

I hope that you have learned something from today's article! If you did, please feel free to leave a comment in the comments section below 💬 Please do the same if you have any questions, remarks or suggestions for improvement. Thank you for reading MachineCurve today and happy engineering! 😎

* * *

## References

Keras Team. (n.d.). _Keras documentation: Callbacks API_. Keras: the Python deep learning API. [https://keras.io/api/callbacks/](https://keras.io/api/callbacks/)

Keras Team. (2020, April 15). _Keras documentation: Writing your own callbacks_. Keras: the Python deep learning API. [https://keras.io/guides/writing\_your\_own\_callbacks/#a-basic-example](https://keras.io/guides/writing_your_own_callbacks/#a-basic-example)

TensorFlow. (n.d.). _Tf.keras.callbacks.ModelCheckpoint_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)

TensorFlow. (n.d.). _Tf.keras.callbacks.TensorBoard_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/TensorBoard](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)

TensorFlow. (n.d.). _Tf.keras.callbacks.EarlyStopping_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

TensorFlow. (n.d.). _Tf.keras.callbacks.LearningRateScheduler_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/LearningRateScheduler](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler)

TensorFlow. (n.d.). _Tf.keras.callbacks.ReduceLROnPlateau_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau)

TensorFlow. (n.d.). _Tf.keras.callbacks.RemoteMonitor_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/RemoteMonitor](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/RemoteMonitor)

TensorFlow. (n.d.). _Tf.keras.callbacks.LambdaCallback_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/LambdaCallback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback)

TensorFlow. (n.d.). _Tf.keras.callbacks.TerminateOnNaN_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/TerminateOnNaN](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TerminateOnNaN)

TensorFlow. (n.d.). _Tf.keras.callbacks.BaseLogger_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/BaseLogger](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/BaseLogger)

TensorFlow. (n.d.). _Tf.keras.callbacks.CSVLogger_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/CSVLogger](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger)

TensorFlow. (n.d.). _Tf.keras.callbacks.History_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/History](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)

TensorFlow. (n.d.). _Tf.keras.callbacks.ProgbarLogger_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ProgbarLogger](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ProgbarLogger)

TensorFlow. (n.d.). _Tf.keras.callbacks.experimental.BackupAndRestore_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/experimental/BackupAndRestore](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/experimental/BackupAndRestore)

TensorFlow. (n.d.). _Tf.keras.callbacks.Callback_. [https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
