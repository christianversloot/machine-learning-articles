---
title: "TensorFlow model optimization: introducing weight clustering"
date: "2020-10-06"
categories: 
  - "frameworks"
tags: 
  - "clustering"
  - "edge-ai"
  - "machine-learning"
  - "tensorflow"
  - "model-optimization"
---

Today's state-of-the-art deep learning models are deep - which means that they represent a large hierarchy of layers which themselves are composed of many weights often. The consequence of their depth is that when saving [model weights](https://www.machinecurve.com/index.php/2019/08/22/what-is-weight-initialization/) after training, the resulting files can become [really big](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/#the-need-for-model-optimization). This poses relatively large storage requirements to hardware where the model runs on. In addition, as running a model after it was trained involves many vector multiplications in [the forward pass of data](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process), compute requirements are big as well.

Often, running such machine learning models in the field is quite impossible due to these resource requirements. This means that cloud-based hardware, such as heavy GPUs, are often necessary to generate predictions with acceptable speed.

Now, fortunately, there are ways to optimize one's model. In other articles, we studied [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/) which changes number representation and [pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/) for zeroing out weights that contribute insignificantly to model performance. However, there is another technique: **weight clustering**. In short, and we shall look into the technique in more detail in this article, it involves reduction of model size by clustering layer weights and subsequently changing the weights that belong to a cluster from their own representation into that of their cluster centroids.

Now, I can imagine that this all sounds a bit abstract. Let's therefore move forward quickly and take a look in more detail. Firstly, we'll cover the need for model optimization - briefly, as we have done this in the articles linked above as well. Secondly, we'll take a look at what weight clustering is conceptually - and why it could work. Then, we cover `tfmot.clustering`, the weight clustering representation available in the TensorFlow Model Optimization Toolkit. Finally, we'll create a Keras model ourselves, and subsequently attempt to reduce its size by applying weight clustering. We also take a look at whether clustering the weights of a pruned and quantized model makes the model even smaller, and what it does to accuracy.

\[toc\]

* * *

## The need for model optimization

We already saw it in the introduction of this article: machine learning models that are very performant these days are often also very big. The reason why is twofold. First of all, after the [2012 deep learning breakthrough](https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/), people found that by making neural networks deeper and deeper, learned representations could be much more complex. Hence, model performance increased while data complexity did too - which is a good thing if you're trying to build models that should work in the real world.

Now, as we saw above, a neural network is essentially a system of neurons, with _model weights_, that are initialized and subsequently optimized. When the neural network is deep, and could potentially be broad as well, the number of so-called _trainable parameters_ is huge! That's the second reason why today's neural networks are very big: their architecture or way of working requires them to be so, when combined with the need for deep networks emerging from the 2012 breakthrough.

When machine learning models are big, it becomes more and more difficult to run them without having dedicated hardware for doing so. In particular, Graphical Processing Units (GPUs) are required if you want to run very big models at speed. Loading the models, getting them to run, and getting them to run at adequate speed - this all gets increasingly difficult when the model gets bigger.

In short, running models in the field is not an easy task today. Fortunately, for the TensorFlow framework, there are methods available for optimizing your neural network. While we covered [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/) and [pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/) in another article, we're going to focus on the third method here today: **weight clustering**.

Let's take a look!

* * *

## Weight clustering for model optimization

Training a neural network is a supervised learning operation: it is trained following the [high-level supervised machine learning process](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process), involving training samples and their corresponding ground truth. However, if you are already involved with Machine Learning, you'll likely also know that there is a branch of techniques that fall under the umbrella of unsupervised learning. [Clustering](https://www.machinecurve.com/index.php/2020/04/16/how-to-perform-k-means-clustering-with-python-in-scikit/) is one of those techniques: without any training samples, an algorithm attempts to identify 'clusters' of similar samples.

![](images/weight_images.jpg)

A representation of model weights in TensorBoard.

They can be used for many purposes - and as we shall see, they can also be used for model optimization by means of clustering weights into groups of similar ones.

### High-level supervised ML process

Identifying how this works can be done by zooming in to the [supervised machine learning process](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process). We know that during training it works by means of a forward pass and subsequent optimization, and that this happens iteratively. In more detail, this is a high-level description of that flow:

- Before the first iteration, weights are [initialized pseudorandomly with some statistical deviation](https://www.machinecurve.com/index.php/2019/09/16/he-xavier-initialization-activation-functions-choose-wisely/).
- In the first iteration, samples are fed forward - often in batches of samples - after which predictions are generated.
- These predictions are compared with ground truth and converge into a _[loss value](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/)_, which is subequently used to [optimize](https://www.machinecurve.com/index.php/2019/10/24/gradient-descent-and-its-variants/) i.e. adapt model weights.
- The iteration is repeated until the preconfigured amount of iterations was completed or a [threshold](https://www.machinecurve.com/index.php/2019/05/30/avoid-wasting-resources-with-earlystopping-and-modelcheckpoint-in-keras/) is met.

This means that after every iteration (i.e. attempt to train the model), weights are adapted. Essentially, this can be characterized as a continuous 'system state change', where the state of the system of weights changes because the weights are adapted. Once training finishes, the state remains constant - until the model is subsequently trained further e.g. with additional data.

### Weight representation

Now, weights themselves are represented mathematically by means of vectors. Those vectors contain numbers given some dimensionality, which can be configured by the ML engineer. All those numbers capture a small part of the learning performed, while the system of numbers (scalars) / vectors as a whole captures all the patterns that were identified in the dataset with respect to the predicted value.

Using blazing-fast mathematical programming libraries, we can subsequently perform many computations at once in order to train the model (i.e. the forward pass) or model inference (generating predictions for new samples, which is essentially also a forward pass, but then without subsequent optimization).

### Clustering weights for model compression benefits

If weights are represented numerically, it is possible to apply [clustering](https://www.machinecurve.com/index.php/2020/04/23/how-to-perform-mean-shift-clustering-with-python-in-scikit/) techniques to them in order to identify groups of similar weights. This is precisely how **weight clustering for model optimization works**. By applying a clustering technique, it is possible to reduce the number of unique weights that are present in a machine learning model (TensorFlow, n.d.).

How this works is as follows. First of all, you need a trained model - where the system of weights can successfully generate predictions. Applying weight clustering based optimization to this model involves grouping the weights of layers into \[latex\]N\[/latex\] clusters, where \[latex\]N\[/latex\] is configurable by the Machine Learning engineer. This is performed using some clustering algorithm (we will look at this in more detail later).

If there's a cluster of samples, it's possible to compute a value that represents the middle of a cluster. This value is called a **centroid** and plays a big role in clustering based model optimization. Here's why: we can argue that the centroid value is the 'average value' for all the weights in the particular cluster. If you remove a bit from one vector in the cluster to move towards the centroid, and add a bit to another cluster, one could argue that - holistically, i.e. from a systems perspective - the model shouldn't lose too much of its predictive power.

And that's precisely what weight clustering based optimization does (TensorFlow, n.d.). Once clusters are computed, all weights in the cluster are adapted to the cluster's centroid value. This brings benefits in terms of model compression: values that are equal can be compressed better. People from TensorFlow have performed tests and have seen up to 5x model compression imrpovements _without_ losing predictive performance in the machine learning model (TensorFlow, n.d.). That's great!

Applying weight clustering based optimization can therefore be a great addition to your existing toolkit, which should include [quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/) and [pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/).

Now that we know what weight clustering based optimization involves, it's time to take a look at how weight clustering based model optimization is implemented in TensorFlow.

* * *

## Weight clustering in the TensorFlow Model Optimization Toolkit

For those who use TensorFlow for creating their neural networks, I have some good news: optimizing machine learning inference is relatively easy, because it can be done with what is known as the [TensorFlow Model Optimization Toolkit, or TFMOT](https://www.tensorflow.org/model_optimization/guide). This toolkit provides functionality for quantization, pruning and weight clustering and works with the Keras models you already created with TensorFlow 2.x.

In this section, we'll be looking at **four components** of weight clustering in TFMOT, namely:

1. **Cluster\_weights(...):** used for wrapping your regular Keras model with weight clustering wrappers, so that clustering can happen.
2. **CentroidInitialization:** used for computation of the initial values of the cluster centroids used in weight clustering.
3. **Strip\_clustering(...):** used for stripping the wrappers off your clustering-ready Keras model, to get back to normal.
4. **Cluster\_scope(...):** used when deserializing (i.e. loading) your weight clustered neural network.

Let's now take a look at each of them in more detail.

### Enabling clustering: cluster\_weights(...)

A regular Keras model cannot be weight clustered as it lacks certain functionality for doing so. That's why we need to _wrap_ the model with this functionality, which clusters weights during training. It is essentially the way to configure weight clustering for your Keras model. Do note, however, as we shall see in the tips later in this article, that you should only cluster a model that already shows acceptable performance e.g. because it was trained before.

Applying `cluster_weights(...)` works as follows (source: [TensorFlow](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering/keras/cluster_weights), license: [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), no changes):

```
clustering_params = {
  'number_of_clusters': 8,
  'cluster_centroids_init':
      CentroidInitialization.DENSITY_BASED
}

clustered_model = cluster_weights(original_model, **clustering_params)
```

Here, we define the number of clusters we want, as well as how the centroids are initialized - a configuration option that we will look at in more detail next. Subsequently, we pass the clustering parameters into `cluster_weights(...)` together with our original model. The `clustered_model` that remains can then be used for clustering.

### Determining centroid initialization: CentroidInitialization

From the section above, we know that weight clustering involves clustering the weights (no shit, sherlock) but then also replacing the weights that are part of a cluster with the centroids of that particular cluster. This achieves the benefits in terms of compression that we talked about.

Understanding that there are multiple [algorithms](https://www.machinecurve.com/index.php/2020/04/23/how-to-perform-mean-shift-clustering-with-python-in-scikit/) for [clustering](https://www.machinecurve.com/index.php/2020/04/16/how-to-perform-k-means-clustering-with-python-in-scikit/) yields the question if certain alterations are present within the TFMOT based weights clustering technique as well.

Now, while it seems to be the case that the _clustering algorithm itself cannot be chosen_ (it seems like [K-means is used under the hood](https://www.machinecurve.com/index.php/2020/04/16/how-to-perform-k-means-clustering-with-python-in-scikit/)), it's possible to choose what is known as a **centroid initialization**. Here's what centroid initialization involves. When starting clustering, as we saw in the previous section, the Machine Learning engineer can configure a number of clusters for either the model or the layers that they intend to cluster.

Those _clusters_ need to be initialized - that is, they need to be placed somewhere in sample space, before the clustering algorithm can work towards convergence. This initial placement is called the initialization of the centers of the clusters, also known as the centroids. In TensorFlow model optimization, a strategy for doing so can be chosen by means of a `CentroidInitialization` parameter. You can choose from the following centroid initialization strategies:

- **Density-based initialization:** using the density of the sample space, the centroids are initialized, as to distribute them with more centroids present in more dense areas of the feature space.
- **Linear initialization:** centroids are initialized evenly spaced between the minimum and maximum weight values present, ignoring any density.
- **Random initialization:** as the name suggests, cluster centroids are chosen by sampling randomly in between the minimum and maximum weight values present.
- **Kmeans++-based initialization:** using Kmeans++, the cluster centroids are initialized.

### Stripping clustering wrappers: strip\_clustering(...)

We know that we had to apply `cluster_weights(...)` in order to wrap the model with special functionality in order to be able to apply clustering in the first place. However, this functionality is no longer required when the model was weight clustered - especially because it's the _weights_ that are clustered, and they belong to the original model.

That's why it's best, and even required, to remove the clustering wrappers if you wish to see the benefits from clustering in terms of reduction of model size when compressed. `strip_clustering(...)` can be used for this purpose. Applying it is really simple: you pass the clustered model, and get a stripped model, like this:

```
model = tensorflow.keras.Model(...)
wrapped_model = cluster_weights(model)
stripped_model = strip_clustering(wrapped_model)
```

### Model deserialization: cluster\_scope(...)

Sometimes, however, you [save a model](https://www.machinecurve.com/index.php/2020/02/14/how-to-save-and-load-a-model-with-keras/) when it is wrapped with clustering functionality:

```
model = tf.Keras.Model(...)
wrapped_model = cluster_weights(model)
tensorflow.keras.models.save_model(wrapped_model, './some_path')
```

If you then [load the model](https://www.machinecurve.com/index.php/2020/02/14/how-to-save-and-load-a-model-with-keras/) with `load_model`, things will go south! This originates from the fact that you are trying to load a _regular_ Keras model, i.e. a model without wrappers, while in fact you saved the model _with_ clustering wrappers.

Fortunately, TFMOT provides functionality to put the loading operation to `cluster_scope` which means that it takes into account the fact that it is loading a model that has been wrapped with clustering functionality:

```
model = tf.Keras.Model(...)
wrapped_model = cluster_weights(model)
file_path = './some_path'
tensorflow.keras.models.save_model(wrapped_model, file_path)

with tfmot.clustering.keras.cluster_scope():
  loaded_model = tensorflow.keras.models.load_model(file_path)
```

* * *

## Tips for applying weight clustering

If you want to apply weight clustering based optimization, it's good to follow a few best practices. Here, we've gathered a variety tips from throughout the web that help you get started with this model optimization technique (TensorFlow, n.d.):

- Weight optimization can be combined with **[post-training quantization](https://www.machinecurve.com/index.php/2020/09/16/tensorflow-model-optimization-an-introduction-to-quantization/)**. This should bring even more benefits compared to weight clustering based optimization or quantization based optimization alone.
- A model should already be trained before weight based clustering is performed. Contrary to e.g. [pruning](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/), where sparsity can be increased while the model is training, weight based pruning does not work in parallel with the training process. It must be applied after training finishes.
- If you apply clustering to layers that precede a batch normalization layer, the benefits are reduced. This is likely due to the normalizing effect of Batch Normalization layers.
- It could be that clustering weights for all layers leads to unacceptable accuracies or other loss scores. In those cases, it is possible to cluster only a few layers only. Click [here](https://www.tensorflow.org/model_optimization/guide/clustering/clustering_comprehensive_guide#cluster_some_layers_sequential_and_functional_models) to find out more if that's what you want to do.
- Apparently, downstream layers (i.e. the later layers in your neural network) have _more redundant parameters_ compared to layers early in the neural network (TensorFlow, n.d.). Here, weight clustering based optimization should provide the biggest benefits. If you want to clusters a few layers only, it could be worthwhile to optimize those later layers instead of early ones.
- Critical layers (e.g. attention layers) should not be clustered; for example, because attention could be lost.
- If you're optimizing a few layers only using weight based optimization, it's important to freeze the first few layers. This ensures that they remain _constant_; if you don't, it could be the case that their weights change in order to accomodate for the changes in the later layers. You often don't want this to happen.
- The way the algorithm computes the centroids of the clusters "plays a key role in (..) model accuracy" (TensorFlow, n.d.). Generally, linear > density based centroid initialization, and linear > random based centeroid initialization. Sometimes, though, the others _are_ better, however only in a minority of cases. Do make sure to test all of them, but if you want to use some heuristics, there they are.
- Fine tuning a model during weight clustering must be done with a learning rate that is lower than the one used in training. This ensures that there won't be any jumpiness in terms of the weights, but that instead the 'optimization steps' performed jointly with clustering are really small.
- If you want to see the compression benefits, you must **both** use `strip_clustering` (which removes the clustering wrappers) and a compression algorithm (such as `gzip`). If you don't, you won't see the benefits.

* * *

## Example: weight clustering your Keras model

Let's now take a step away from all the theory - we're going to code a model that applies weight clustering based optimization for a Keras model 😎

### Defining the ConvNet

For this example, we're going to create a simple [Convolutional Neural Network](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/) with Keras that is trained to recognize digits from the MNIST [dataset](https://www.machinecurve.com/index.php/2019/12/31/exploring-the-keras-datasets/). If you're familiar with Machine Learning, you're well aware that this dataset is used in educational settings very often. Precisely that is the reason that we are also using this dataset here today. In fact, it's a model that will _guarantee_ to perform well (if trained adequately), often with accuracies of 95-97% and more.

Do note that if you wish to run the model code, you will need `tensorflow` 2.x as well as the TensorFlow Model Optimization Toolkit or `tfmot`. If you don't have it already, you must also install NumPy. Here's how to install them:

- **TensorFlow:** `pip install tensorflow`
- **TensorFlow Model Optimization Toolkit:** `pip install --upgrade tensorflow-model-optimization`
- **NumPy:** `pip install numpy`

![](images/mnist.png)

Samples from the MNIST dataset.

### Compiling and training the ConvNet

The first step is relatively simple, and we'll skip the explanations for this part. If you don't understand them yet but would like to do so, I'd recommend clicking the link to the ConvNet article above, where I explain how this is done.

Now, open up some file editor, create a file - e.g. `clustering.py`. It's also possible to use a Jupyter Notebook for this purpose. Then, add this code, which imports the necessary functionality, defines the architecture for our neural network, compiles it and subsequently fits it i.e. starts the training process:

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
no_epochs = 15

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
```

### Generating evaluation metrics for the ConvNet

After fitting the data to your model, you have exhausted your training set _and_ your validation dataset. That is, you can't use both datasets in order to test how well it performs - because both have played a role in the training process.

You don't want to be the butcher who checks their own meat, don't you?

Instead, in the code above, we have split off a part of the dataset (in fact, Keras did that for us) which we can use for _testing_ purposes. It allows us to test how well our model performs when it is ran against samples it hasn't seen before. In ML terms, we call this _testing how well the model generalizes._

With Keras, you can easily evaluate model performance:

```
# Generate generalization metrics for original model
score = model.evaluate(input_test, target_test, verbose=0)
```

### Storing the ConvNet to file

Later in this article, we're going to compare the size of a compressed-and-saved model that was optimized with weights clustering to the size of a compressed-and-saved original model. If we want to do this, we must save the original model to a temporary file. Here's how we do that, so let's add this code next:

```
# Store file 
_, keras_file = tempfile.mkstemp('.h5')
save_model(model, keras_file, include_optimizer=False)
print(f'Baseline model saved: {keras_file}')
```

### Configuring weight clustering for the ConvNet

Now that we have trained, evaluated and saved the original ConvNet, we can move forward with the actual weights clustering related operations. The first thing we're going to do is configuring how TensorFlow will cluster weights during finetuning.

For this reason, we're going to create a dictionary with the `number_of_clusters` we want the clustering algorithm to find and how the cluster centroids are initialized:

```
# Define clustering parameters
clustering_params = {
  'number_of_clusters': 14,
  'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
}
```

We want 14 clusters. In line with the tips from above, we're using a `CentroidInitialization.LINEAR` strategy for applying weight clustering here.

### Compiling and finetuning the clustered model

Then, it's time to wrap our trained `model` with clustering functionality configured according to our `clustering_params`:

```
# Cluster the model
wrapped_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)
```

We're now almost ready to finetune our model with clustered weights. However, recall from the tips mentioned above that it is important to decrease the learning rate when doing so. That's why we're redefining our [Adam optimizer](https://www.machinecurve.com/index.php/2019/11/03/extensions-to-gradient-descent-from-momentum-to-adabound/) with a lower learning rate (`1e-4` by default):

```
# Decrease learning rate (see tips in article!)
decreased_lr_optimizer = tensorflow.keras.optimizers.Adam(lr=1e-5)
```

We then recompile the model and finetune _for just one epoch_:

```
# Compile wrapped model
wrapped_model.compile(
  loss=tensorflow.keras.losses.categorical_crossentropy,
  optimizer=decreased_lr_optimizer,
  metrics=['accuracy'])

# Finetuning
wrapped_model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=1,
          verbose=verbosity,
          validation_split=validation_split)
```

### Evaluating the clustered model

Here, too, we must investigate how well the clustered model generalizes. We add the same metrics _and also print the outcomes of the previous evaluation step:_

```
# Generate generalization metrics for clustered model
clustered_score = model.evaluate(input_test, target_test, verbose=0)
print(f'Regular CNN - Test loss: {score[0]} / Test accuracy: {score[1]}')
print(f'Clustered CNN - Test loss: {clustered_score[0]} / Test accuracy: {clustered_score[1]}')
```

### Comparing the clustered and original models

For comparing the clustered and original models, we must do a few things:

1. Remember to use `strip_clustering(...)` in order to convert our wrapped model back into a regular Keras model.
2. Store our file.
3. Gzip both of our models, and run our example.

First of all, we strip the wrappers and store our file:

```
# Strip clustering
final_model = tfmot.clustering.keras.strip_clustering(wrapped_model)

# Store file 
_, keras_file_clustered = tempfile.mkstemp('.h5')
save_model(final_model, keras_file_clustered, include_optimizer=False)
print(f'Clustered model saved: {keras_file_clustered}')
```

Then, we're using a Python definition provided by TensorFlow (Apache 2.0 licensed) to get the size of our gzipped model:

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
```

The last thing is comparing the sizes of both models when compressed:

```
print("Size of gzipped original Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped clustered Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file_clustered)))
```

* * *

## Running the example

Time to run the example!

Open up your Python example, such as your terminal or your Notebook, and run the code - e.g. with `python clustering.py`. You will likely observe the following:

1. Your model will train for 15 epochs, and will achieve significantly low loss scores and high accuracies relatively soon - it's the MNIST dataset, after all.
2. Your model will then train for 1 epoch, and likely, this will be significantly slower than each of the 15 epochs (remember that clustering is applied here under the hood).
3. Your model will then print both the evaluation and the compression comparison scores.

In my case, this produced the following numbers:

```
Regular CNN - Test loss: 0.02783038549570483 / Test accuracy: 0.9919999837875366
Clustered CNN - Test loss: 0.027621763848347473 / Test accuracy: 0.9919000267982483
Size of gzipped original Keras model: 1602422.00 bytes
Size of gzipped clustered Keras model: 196180.00 bytes
```

We see a reduction in size of **more than 8 times** with a _very small loss of performance_. That's awesome! 😎

* * *

## Summary

Today's machine learning models can become very large, hampering things like model inference in the field. Another factor that is impacted is storage: weights must both be stored and loaded, impacting performance of your Edge AI scenario and incurring additional costs.

Fortunately, with modern machine learning libraries like TensorFlow, it is possible to apply a variety of optimization techniques to your trained ML models. In another posts, we focused on quantization and pruning. In this article, we looked at weights clustering: the application of an unsupervised clustering algorithm to cluster the weights of your machine learning model in \[latex\]N\[/latex\] clusters. How this optimizes your machine learning model is relatively easy: as weights within the clusters are set to the centroid values for each cluster, model compression benefits are achieved, as the same numbers can be comrpessed more easily.

In the remainder of the article, we specifically looked at how weight clustering based model optimization is presented within the API of the TensorFlow Model Optimization Toolkit. We looked at how Keras models can be wrapped with clustering functionality, what initialization strategies for the cluster centroids can be used, how models can be converted back into regular Keras models after training and finally how wrapped models can be deserialized.

We extended this analysis by means of an example, where we trained a simple Keras CNN on the MNIST dataset and subsequently applied weight clustering. We noticed that the size of our compressed Keras model was reduced by more than 8 times with only a very small reduction in performance. Very promising indeed!

I hope that you have learnt a lot from this article - I did, when researching :) Please feel free to leave a message if you have any remarks, questions or other suggestions for the improvement of this post. If not, thanks for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

* * *

## References

_Module: Tfmot.clustering_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/clustering](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering)

_Module: Tfmot.clustering.keras_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/clustering/keras](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering/keras)

_Module: Tfmot.clustering.keras_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/clustering/keras](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering/keras)

_Tfmot.clustering.keras.CentroidInitialization_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/clustering/keras/CentroidInitialization](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering/keras/CentroidInitialization)

_Tfmot.clustering.keras.cluster\_scope_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/clustering/keras/cluster\_scope](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering/keras/cluster_scope)

_Tfmot.clustering.keras.cluster\_weights_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/clustering/keras/cluster\_weights](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering/keras/cluster_weights)

_Tfmot.clustering.keras.strip\_clustering_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/clustering/keras/strip\_clustering](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/clustering/keras/strip_clustering)

_Weight clustering in Keras example_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/guide/clustering/clustering\_example](https://www.tensorflow.org/model_optimization/guide/clustering/clustering_example)

_Weight clustering_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/guide/clustering](https://www.tensorflow.org/model_optimization/guide/clustering)
