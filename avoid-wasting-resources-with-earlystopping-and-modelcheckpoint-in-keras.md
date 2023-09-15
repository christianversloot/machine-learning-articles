---
title: "Using EarlyStopping and ModelCheckpoint with TensorFlow 2 and Keras"
date: "2019-05-30"
categories:
  - "buffer"
  - "deep-learning"
  - "frameworks"
tags:
  - "ai"
  - "callbacks"
  - "deep-learning"
  - "keras"
  - "neural-networks"
---

Training a neural network can take a lot of time. In some cases, especially with very deep architectures trained on very large data sets, it can take weeks before one's model is finally trained.

In Keras, when you train a neural network such as a [classifier](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/) or a [regression model](https://www.machinecurve.com/index.php/2019/07/30/creating-an-mlp-for-regression-with-keras/), you'll usually set the number of epochs when you call `model.fit`:

```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
```

Unfortunately, setting a fixed number of epochs is often a **bad idea**. Here's why:

- When you use too few epochs, your model will remain underfit. What I mean is that its predictive power can still be improved without a loss of generalization power (i.e., it improves without overfitting). You will end up with a model that does not perform at its maximum capability.
- When you use too many epochs, depending on how you configure the training process, your final model will either be _optimized_ or it will be _overfit_. In both cases, you will have wasted resources. Hey, but why are those resources wasted when the final model is optimal? Simple - most likely, this optimum was found in e.g. 20% of the epochs you configured the model for. 80% of the resources you used are then wasted. Especially with highly expensive tasks in computational terms, you'll want to avoid waste as much as you can.

This is quite a dilemma, isn't it? How do we choose what number of epochs to use?

You cannot simply enter a random value due to the reasons above.

Neither can you test without wasting more resources. What's more, if you think to avert the dilemma by finding out with a very small subset of your data, then I've got some other news - you just statistically altered your sample by drawing a subset from the original sample. You may now find that by using the original data set for training, it is still not optimal.

What to do? :( In this tutorial, we'll check out one way of getting beyond this problem: using a combination of **Early Stopping** and **model checkpointing**. Let's see what it is composed of.

In other words, this tutorial will teach you...

- **Why performing early stopping and model checkpointing can be beneficial.**
- **How early stopping and model checkpointing are implemented in TensorFlow.**
- **How you can use `EarlyStopping` and `ModelCheckpoint` in your own TensorFlow/Keras model.**

Let's take a look 🚀

* * *

**Update 13/Jan/2021:** Added code example to the top of the article, so that people can get started immediately. Also ensured that the article is still up-to-date, and added a few links to other articles.

**Update 02/Nov/2020:** Made model code compatible with TensorFlow 2.x.

**Update 01/Feb/2020:** Added links to other MachineCurve blog posts and processed textual corrections.

* * *

\[toc\]

* * *

## Code example: how to use EarlyStopping and ModelCheckpoint with TensorFlow?

This code example immediately teaches you **how EarlyStopping and ModelCheckpointing can be used with TensorFlow**. It allows you to get started straight away. If you want to understand both callbacks in more detail, however, then make sure to continue reading the rest of this tutorial.

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2, 
          callbacks=keras_callbacks)
```

* * *

## EarlyStopping and ModelCheckpoint in Keras

Fortunately, if you use Keras for creating your deep neural networks, it comes to the rescue.

It has two so-called [callbacks](https://www.machinecurve.com/index.php/mastering-keras/#keras-callbacks) which can really help in settling this issue, avoiding wasting computational resources a priori and a posteriori. They are named `EarlyStopping` and `ModelCheckpoint`. This is what they do:

- **EarlyStopping** is called once an epoch finishes. It checks whether the [metric](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/) you configured it for has improved with respect to the best value found so far. If it has not improved, it increases the count of 'times not improved since best value' by one. If it did actually improve, it resets this count. By configuring your _patience_ (i.e. the number of epochs without improvement you allow before training should be aborted), you have the freedom to decide when to stop training. This allows you to configure a very large number of epochs in model.fit (e.g. 100.000), while you know that it will abort the training process once it no longer improves. Gone is your waste of resources with respect to training for too long.
- It would be nice if you could save the best performing model automatically. **ModelCheckpoint** is perfect for this and is also called after every epoch. Depending on how you configure it, it saves the entire model or its weights to an HDF5 file. If you wish, it can only save the model once it has improved with respect to some metric you can configure. You will then end up with the best performing instance of your model saved to file, ready for loading and production usage.

Together, EarlyStopping and ModelCheckpoint allow you to stop early, saving computational resources, while maintaining the best performing instance of your model automatically. That's precisely what you want.

* * *

## Example implementation

Let's build one of the [Keras examples](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py) step by step. It uses one-dimensional [convolutional layers](https://machinecurve.com/index.php/2018/12/07/convolutional-neural-networks-and-their-components-for-computer-vision/) for classifying IMDB reviews and, according to its metadata, achieves about 90% test accuracy after just two training epochs.

We will slightly alter it in order to (1) include the callbacks and (2) keep it running until it no longer improves.

Let's first load the Keras imports. Note that we also include `numpy`, which is not done in the Keras example. We include it because we'll need to fix the random number generator, but we'll come to that shortly.

```python
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.datasets import imdb
import numpy as np
```

We will then set the parameters. Note that instead of 2 epochs in the example, we'll use 200.000 epochs here.

```python
# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 200000
```

We'll fix the random seed in Numpy. This allows us to use the same pseudo random number generator every time. This removes the probability that variation in the data is caused by the pseudo-randomness between multiple instances of a 'random' number generator - rather, the pseudo-randomness is equal all the time.

```python
np.random.seed(7)
```

We then load the data. We make a `load_data` call to the [IMDB data set](https://www.machinecurve.com/index.php/2019/12/31/exploring-the-keras-datasets/#imdb-movie-reviews-sentiment-classification), which is provided in Keras by default. We load a maximum of 5.000 words according to our configuration file. The `load_data` definition provided by Keras automatically splits the data in training and testing data (with inputs `x` and targets `y`). In order to create feature vectors that have the same shape, the sequences are padded. That is, `0.0` is added towards the end. Neural networks tend not to be influenced by those numbers.

```python
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
```

Next up is the model itself. It is proposed by Google. Given the goal of this blog post, there's not much need for explaining whether the architecture is good (which is the case, though):

```python
model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```

Next, we compile the model. [Binary crossentropy](https://www.machinecurve.com/index.php/2019/10/22/how-to-use-binary-categorical-crossentropy-with-keras/) is used since we have two target classes (`positive` and `negative`) and our task is a classification task (for which [crossentropy](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#binary-crossentropy) is a good way of computing loss). The optimizer is [Adam](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/#adam), which is a state-of-the-art optimizer combining various improvements to original [stochastic gradient descent](https://www.machinecurve.com/index.php/2019/10/24/gradient-descent-and-its-variants/). As an additional metric which is more intuitive to human beings, `accuracy` is included as well.

```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

We'll next make slight changes to the example. Google utilizes the `test` data for validation; we don't do that. Rather, we'll create a separate validation split from the training data. We thus end up with three distinct data sets: a training set, which is used to train the model; a validation set, which is used to study its predictive power after every epoch, and a testing set, which shows its generalization power since it contains data the model has never seen. We generate the validation data by splitting the training data in actual training data and validation date. We use a 80/20 split for this; thus, 20% of the original training data will become validation data. All right, let's fit the training data and start the training process.

```python
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
```

Later, we'll evaluate the model with the test data.

### Adding the callbacks

We must however first add the callbacks to the imports at the top of our code:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

We can then include them into our code. Just before `model.fit`, add this Python variable:

```python
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
]
```

As you can see, the callbacks have various configuration options:

- The **checkpoint\_path** in ModelCheckpoint is the path to the file where the model instance should be saved. In my case, the checkpoint path is `checkpoint_path=f'{os.path.dirname(os.path.realpath(__file__))}/testmodel.h5'`.
- A **monitor**, which specifies the variable that is being monitored by the callback for making its decision whether to stop or save the model. Often, it's a good idea to use `val_loss`, because it overfits much slower than training loss. This does however require that you add a `validation_split` in `model.fit`.
- A **patience**, which specifies how many epochs without improvement you'll allow before the callback interferes. In the case of EarlyStopping above, once the [validation loss](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/) improves, I allow Keras to complete 30 new epochs without improvement before the training process is finished. When it improves at e.g. the 23rd epoch, this counter is reset and the cycle starts again.
- The **mode**, which can also be `max` or left empty. If it's left empty, it decides itself based on the `monitor` you specify. Common sense dictates what mode you should use. Validation loss should be minimized; that's why we use `min`. Not sure why you would attempt to maximize validation loss :)
- The **min\_delta** in EarlyStopping. Only when the improvement is higher than this delta value it is considered to be an improvement. This avoids that very small improvements disallow you from finalizing training, e.g. when you're trapped in a small convergence scenario when using a really small learning rate.
- The **save\_best\_only** in ModelCheckpoint pretty much speaks for itself. If `True`, it only saves the best model instance with respect to the monitor specified.
- If you wish, you can add `verbose=1` to both callbacks. This textually shows you whether the model has improved or not and whether it was saved to your `checkpoint_path`. I leave this up to you as it slows down the training process slightly (...since the prints must be handled by Python).

Those are not the only parameters. There's many more for both [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) and [EarlyStopping](https://keras.io/callbacks/#earlystopping), but they're used less commonly. Do however check them out!

All right, if we would now add the callback variable to the `model.fit` call, we'd have a model that stops when it no longer improves _and_ saves the best model. Replace your current code with this:

```python
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2, 
          callbacks=keras_callbacks)
```

Okay, let's run it and see what happens :)

![](images/bookshelves-chair-desk-1546912.jpg)

All right, let's give it a go!

* * *

## Numpy allow\_pickle error

It may be that you'll run into issues with Numpy when you load the data into a Numpy array. Specifically, the error looks as follows:

```shell
ValueError: Object arrays cannot be loaded when allow_pickle=False
```

It occurs because Numpy has recently inverted the default value for allow\_pickle and Keras has not updated yet. Altering `imdb.py` in `keras/datasets` folder will resolve this issue. Let's hope the pull request that has been issued for this problem will be accepted soon. Change line 59 into:

```python
with np.load(path, allow_pickle=True) as f:
```

**Update February 2020:** this problem should be fixed in any recent Keras version! 🎉

* * *

## Keras results

You'll relatively quickly see the results:

```shell
Epoch 1/200000
20000/20000 [==============================] - 10s 507us/step - loss: 0.4380 - acc: 0.7744 - val_loss: 0.3145 - val_acc: 0.8706
Epoch 00001: val_loss improved from inf to 0.31446, saving model to C:\Users\chris\DevFiles\Deep Learning/testmodel.h5

Epoch 2/200000
20000/20000 [==============================] - 7s 347us/step - loss: 0.2411 - acc: 0.9021 - val_loss: 0.2719 - val_acc: 0.8890
Epoch 00002: val_loss improved from 0.31446 to 0.27188, saving model to C:\Users\chris\DevFiles\Deep Learning/testmodel.h5

Epoch 3/200000
20000/20000 [==============================] - 7s 344us/step - loss: 0.1685 - acc: 0.9355 - val_loss: 0.2733 - val_acc: 0.8924
Epoch 00003: val_loss did not improve from 0.27188
```

Apparently, the training process achieves optimal validation loss after just two epochs (which was also indicated by the Google engineers who created the model code we are thankful for using and which we adapted), because after epoch 32 it shows:

```shell
Epoch 32/200000
20000/20000 [==============================] - 7s 366us/step - loss: 0.0105 - acc: 0.9960 - val_loss: 0.7375 - val_acc: 0.8780
Epoch 00032: val_loss did not improve from 0.27188
Epoch 00032: early stopping
```

...and the training process comes to a halt, as we intended :) Most likely, the model can still be improved - e.g. by introducing [learning rate decay](https://www.machinecurve.com/index.php/2019/11/11/problems-with-fixed-and-decaying-learning-rates/) and finding the best [learning rate](https://www.machinecurve.com/index.php/2019/11/06/what-is-a-learning-rate-in-a-neural-network/) prior to the training process - but hey, that wasn't the goal of this exercise.

I've also got my HDF5 file:

![](images/image-1.png)

* * *

## Let's evaluate the model

We can next comment out everything from `model = Sequential()` up to and including `model.fit`. Let's add some evaluation functionality.

We should load the model, so we should add its feature to the imports:

```python
from tensorflow.keras.models import load_model
```

And subsequently add evaluation code just after the code that was commented out:

```python
model = load_model(checkpoint_path)
scores = model.evaluate(x_test, y_test, verbose=1)
print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
```

Next, run it again. Instead of training the model again (you commented out the code specifying the model and the training process), it will now load the model you saved during training and evaluate it. You will most likely see a test accuracy of ≈ 88%.

```shell
25000/25000 [==============================] - 3s 127us/step
Score: loss of 0.27852724124908446; acc of 88.232%
```

All right! Now you know how you can use the EarlyStopping and ModelCallback checkpoints in Keras, allowing you to save precious resources when a model no longer improves. Let me wish you all the best with your machine learning adventures and please, feel free to comment if you have questions or comments. I'll be happy to respond and to improve my work if you feel I've made a mistake. Thanks!

* * *

## References

- [Keras,](https://github.com/keras-team/keras) which is licensed under the [MIT License](https://github.com/keras-team/keras/blob/master/LICENSE).
- The specific Keras [example](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py) which lays at the foundation of our blog.
- The [Keras](https://keras.io/) docs.

Thanks a lot to the authors of those works!
