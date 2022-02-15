---
title: "TensorFlow model optimization: an introduction to Quantization"
date: "2020-09-16"
categories: 
  - "deep-learning"
  - "frameworks"
tags: 
  - "edge-ai"
  - "latency"
  - "optimizer"
  - "quantization"
  - "storage"
  - "tensorflow"
  - "model-optimization"
  - "tflite"
---

Since the 2012 breakthrough in machine learning, spawning the hype around deep learning - that should have mostly passed by now, favoring more productive applications - people around the world have worked on creating machine learning models for pretty much everything. Personally, to give an example, I have spent time creating a machine learning model for recognizing the material type of underground utilities using ConvNets for my master's thesis. It's really interesting to see how TensorFlow and other frameworks, such as Keras in my case, can be leveraged to create powerful AI models. Really fascinating!

Despite this positivity, critical remarks cannot be left out. While the research explosion around deep learning has focused on finding alternatives to common loss functions, the effectiveness of Batch Normalization and Dropout, and so on, practical problems remain huge. One class of such practical problems is related to deploying your model in the real world. During training, and especially if you use one of the more state-of-the-art model architectures, you'll create a _very big model_.

Let's repeat this, but then in bold: **today's deep learning models are often very big**. Negative consequences of model size are that very powerful machines are required for inference (i.e. generating predictions for new data) or to even get them running. Until now, those machines have been deployed in the cloud. In situations where you want to immediately respond in the field, creating a cloud connection is not the way to go. That's why today, a trend is visible where machine learning models are moving to the edge. There is however nobody who runs very big GPUs in the field, say at a traffic sign, to run models. Problematic!

Unless it isn't. Today, fortunately, many deep learning tools have built-in means to optimize machine learning models. TensorFlow and especially the TensorFlow Lite set of tools provide many. In this blog, we'll cover **quantization**, effectively a means to reduce the size of your machine learning model by rounding `float32` numbers to nearest smaller-bit ones.

* * *

\[toc\]

* * *

## AI at the edge: the need for model optimization

Let's go back to the core of my master's thesis that I mentioned above - the world of underground utilities. Perhaps, you have already experienced outages some times, but in my country - the Netherlands - things go wrong _once every three minutes_. With 'wrong', I mean the occurrence of a utility strike. Consequences are big: annually, direct costs are approximately 25 million Euros, with indirect costs maybe ten to fifteen times higher.

Often, utility strikes happen because information about utilities present in the underground is outdated or plainly incorrect. Because of this reason, there are companies today which specialize in scanning and subsequently mapping those utilities. For this purpose, among others, they use a device called a _ground penetrating radar_ (GPR). Using a GPR, which emits radio waves into the ground and subsequently stores the reflections, geophysicists scan and subsequently generate maps of what's subsurface.

Performing such scans and generating those maps is a tedious task. First of all, the engineers have to walk hundreds of meters to perform the scanning activities. Subsequently, they must scrutinize all those hundreds of metres - often in a repetitive way. Clearly, this presents opportunities for automation. And that's what I attempted to do in my master's thesis: amplify the analyst's knowledge by using machine learning - and specifically today's ConvNets - to automatically classify objects on GPR imagery with respect to radar size.

https://www.youtube.com/watch?v=oQaRfA7yJ0g

While very interesting from a machine learning point of view, that should not be the end goal commercially. The holy grail would be to equip a GPR device with a machine learning model that is very accurate and which generalizes well. When this happens, engineers who dig in the underground can _themselves_ perform those scan, and subsequently _analyze themselves where they have to be cautious_. What an optimization that would be compared to current market conditions, which are often unfavorable for all parties involved.

Now, if that would be the goal, we'd have to literally _run_ the machine learning model on the GPR device as well. That's where we repeat what we discussed in the beginning of this blog: given the sheer size of today's deep learning models, that's practically impossible. Nobody will equip a hardware device used in the field with a very powerful GPU. And if they would, where would they get electricity from? It's unlikely that it can be powered by a simple solar panel.

Here emerges the need for creating machine learning models that run in the field. In business terms, we call this **Edge AI** - indeed, AI is moving from centralized orchestrations in clouds to the edge, where it can be applied instantly and where insights can be passed to actuators immediately. But doing so requires that models become efficient - much more efficient. Fortunately, many frameworks - TensorFlow included - provide means for doing so. Next, we'll cover TensorFlow Lite's methods for optimization related to **quantization**. Other optimization methods, such as pruning, will be discussed in future blogs.

* * *

## Introducing Quantization

Optimizing a machine learning model can be beneficial in multiple ways (TensorFlow, n.d.). Primarily, **size reduction**, **latency reduction** and **accelerator compatibility** can be reasons to optimize one's machine learning model. With respect to reducing _model size_, benefits are as follows (TensorFlow, n.d.):

> **Smaller storage size:** Smaller models occupy less storage space on your users' devices. For example, an Android app using a smaller model will take up less storage space on a user's mobile device.
> 
> **Smaller download size:** Smaller models require less time and bandwidth to download to users' devices.
> 
> **Less memory usage:** Smaller models use less RAM when they are run, which frees up memory for other parts of your application to use, and can translate to better performance and stability.

That's great from a cost perspective, as well as a user perspective. The benefits of _latency reduction_ compound this effect: because the model is smaller and more efficient, it takes less time to let a new sample pass through it - reducing the time between generating a prediction and _receiving_ that prediction. Finally, with respect to _accelerator compatibility_, it's possible to achieve extremely good results when combining optimization with TPUs, which are specifically designed to run TensorFlow models (TensorFlow, n.d.). Altogether, optimization can greatly increase machine learning cost performance while keeping model performance at similar levels.

### Float32 in your ML model: why it's great

By default, TensorFlow (and Keras) use `float32` number representation while training machine learning models:

```
>>> tf.keras.backend.floatx()
'float32'
```

Floats or _floating-point numbers_ are "arithmetic using formulaic representation of real numbers as an approximation to support a trade-off between range and precision. For this reason, floating-point computation is often found in systems which include very small and very large real numbers, which require fast processing times" (Wikipedia, 2001). Put plainly, it's a way of representing _real_ numbers (a.k.a., numbers like 1.348348348399...), ensuring processing speed, while having only a minor trade-off between range and precision. This is contrary to _integer_ numbers, which can only be round (say, 10, or 3, or 52).

Floats can always store a number of _bits_, or 0/1 combinations. The same is true for integers. The number after `float` in `float32` represents the number of bits with which Keras works by default: 32. Therefore, it works with 32-bit floating point numbers. As you can imagine, such `float32`s can store significantly more precise data compared to `int32` - it can represent 2.12, for example, while `int32` can only represent 2, and 3. That's the first benefit of using floating point systems in your machine learning model.

This directly translates into another benefit of using `float`s in your deep learning model. Training your machine learning process is continuous (Stack Overflow, n.d.). This means that weight initialization, backpropagation and subsequent model optimization - a.k.a. [the high-level training process](https://www.machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions/#the-high-level-supervised-learning-process) - benefits from _very precise numbers_. Integers can only represent numbers between `X` and `Y`, such as 2 and 3. Floats can represent any real number in between the two. Because of this, computations during training can be much more precise, benefiting the performance of your model. Using floating point numbers is therefore great during training.

### Float32 in your ML model: why it's not so great

However, if you want to deploy your model, the fact that it was trained using `float32` is not so great. The precision that benefits the training process comes at a cost: the cost of storing that precision. For example, compared to the integer 3000, 3000.1298289 requires much bigger number systems in order to be represented. This, in return, makes your model bigger and less efficient during inference.

### What model quantization involves

Quantization helps solve this problem. Following TensorFlow (n.d.), "\[it\] works by reducing the precision of the numbers used to represent a model's parameters". Hence, we simply cut off the precision - that is, from 321.36669 to 321. Hoping that the difference wouldn't impact the model in a major way, we can cut model size significantly. In the blog post ["_Here’s why quantization matters for AI_."](https://www.qualcomm.com/news/onq/2019/03/12/heres-why-quantization-matters-ai), Qualcomm (2020) greatly demonstrates why quantization helps reduce the size of your model through quantization by means of an example:

- In order to represent 3452.3194 in floating point numbers, you would need a 32-bit float, thus `float32`.
- Quantizing that number to 3452 however requires an 8-bit integer only, `int8`, which means that you can reserve 24 fewer bits for representing the approximation of that float!

Now that we know what quantization is and how it benefits model performance, it's time to take a look at the quantization approaches supported by TensorFlow Lite. TF Lite is a collection of tools used for model optimization (TensorFlow lite guide, n.d.). It can be used sequential to regular TensorFlow to reduce size and hence increase efficiency of your trained TF models; it can also be installed on edge devices to run the optimized models. TF Lite supports the following methods of quantization:

- **Post-training float16 quantization**: quantizing of model weights and activations from `float32` to `float16`.
- **Post-training dynamic range quantization**: quantizing of model weights and activaitons from `float32` to `int8`. On inference, weights are dequantized back into `float32` (TensorFlow, n.d.).
- **Post-training integer quantization**: converting `float32` activations and model weights into `int8` format. For this reason, it is also called _full integer quantization_.
- **Post-training integer quantization with int16 activations**, also called _16x8 quantization_, allows you to quantize `float32` weights and activations into `int8` and `int16`, respectively.
- **Quantization-aware training:** here, the model is made aware of subsequent quantization activities during training, emulating inference-time quantization during the training process.

### Post-training float16 quantization

One of the simplest quantization approaches is to convert the model's `float32` based weights into `float16` format (TensorFlow, n.d.). This effectively means that the size of your model is reduced by 50%. While the reduction in size is lower compared to other quantization methods (especially the `int` based ones, as we will see), its benefit is that your models will still run on GPUs - and will run faster, most likely. This does however not mean that they don't run on CPUs instead (TensorFlow, n.d.).

- More information about `float16` quantization: [Post-training float16 quantization](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

### Post-training dynamic range quantization

It's also possible to quantize dynamically - meaning that model weights get quantized into `int8` format from `float32` format (TensorFlow, n.d.). This means that your model will become 4 times smaller, or 25% of the original size - a 2x increase compared to post-training `float16` quantization discussed above. What's more, model activations can be quantized as well, but only during inference time.

While models get smaller with dynamic range quantization, you lose the possibility of running your model for inference on a GPU or TPU. Instead, you'll have to use a CPU for this purpose.

- More information about dynamic range quantization: [Post-training dynamic range quantization](https://www.tensorflow.org/lite/performance/post_training_quant)

### Post-training integer quantization (full integer quantization)

Another, more thorough approach to quantization, is to convert "all model math" into `int` format. More precisely, everything from your model is converted from `float32` into `int8` format (TensorFlow, n.d.). This also means that the activations of your model are converted into `int` format, compared to dynamic range quantization, which does so during inference time only. This method is also c alled "full integer quantization".

Integer quantization helps if you want to run your model on a CPU or even an Edge TPU, which requires integer operations in order to accelerate model performance. What's more, it's also likely that you'll have to perform integer quantization when you want to run your model on a microcontroller. Still, despite the model getting smaller (4 times - from 32-bit into 8-bit) and faster, you'll have to think carefully: changing floats into ints removes _precision_, as we discussed earlier. Do you accept the possibility that model performance is altered? You'll have to thoroughly test this if you use integer quantization.

- More information about full integer quantization: [Post-training integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

### Post-training integer quantization with int16 activations (16x8 integer quantization)

Another approach that actually extends the former is post-training integer quantization with int16 activations (TensorFlow, n.d.). Here, weights are converted into `int8`, but activations are converted into `int16` format. Compared to the former, this method is often called "16x8 integer quantization" (TensorFlow, n.d.). It has the benefit that model size is still reduced - because weights are still in `int8` format. However, for inference, greater accuracy is achieved compared to full integer quantization through activation quantization into `int16` format.

- More information about 16x8 integer quantization: [Post-training integer quantization with int16 activations](https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8)

### Quantization-aware training

All previous approaches to quantization require you to train a full `float32` model first, after which you apply one of the forms of quantization to optimize the model. While this is easier, model accuracy could possibly benefit when the model is already aware during training that it will eventually be quantized using one of the quantization approaches discussed above. Quantization-aware training allows you to do this (TensorFlow, n.d.), by emulating inference-time quantization during the fitting process. Doing so allows your model to learm parameters that are _robust_ against the loss of precision invoked with quantization (Tfmot.quantization.keras.quantize\_model, n.d.).

Generally, quantization-aware training is a three-step process:

1. Train a regular model through `tf.keras`
2. Make it quantization-aware by applying the related API, allowing it to learn those loss-robust parameters.
3. Quantize the model use one of the approaches mentioned above.

- More information about quantization-aware training: [Quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training.md)

### Which quantization method to choose for your ML model

It can be difficult to choose a quantization method for your machine learning model. The table below suggests which quantization method could be best for your use case. It seems to be a trade-off between _benefits_ and _hardware_, primarily. Here are some general heuristics:

- If you want to run your quantized model on a GPU, you must use `float16` quantization.
- If you want to benefit from greatest speedups, full `int8` quantization is best.
- If you want to ensure that model performance does not deteriorate significantly when performing full int quantization, you could choose 16x8 quantization instead. On the downside, models remain a bit bigger, and speedups seem to be a bit lower.
- If you're sure that you want to run your model on a CPU, dynamic range quantization is likely useful to you.
- Quantization-aware training is beneficial prior to performing quantization.

| Technique | Benefits | Hardware |
| --- | --- | --- |
| Dynamic range quantization | 4x smaller, 2x-3x speedup | CPU |
| Full integer quantization | 4x smaller, 3x+ speedup | CPU, Edge TPU, Microcontrollers |
| 16x8 integer quantization | 3-4x smaller, 2x-3x+ speedup | CPU, possibly Edge TPU, Microcontrollers |
| Float16 quantization | 2x smaller, GPU acceleration | CPU, GPU |

Benefits of optimization methods and hardware that supports it. Source: [TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization), licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), no changes were made except for the new 16x8 integer quantization row.

* * *

## An example: model quantization for a Keras model

Let's now implement (dynamic range) quantization for a model trained with `tf.keras`, to give an example - and to learn myself as well :) For this, we'll be using a relatively straight-forward [ConvNet created with Keras](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/) that is capable of classifying the MNIST dataset. It allows us to focus on the new aspects of quantization rather than having to worry about how the original neural network works.

### CNN classifier code

Here's the full code for the CNN classifier which serves as our starting point. It constructs a two-Conv-layer neural network combined with [max pooling](https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/) and [Dropout](https://www.machinecurve.com/index.php/2019/12/18/how-to-use-dropout-with-keras/). It trains on the MNIST dataset, which is first converted from `uint8` format into `float32` format - precisely because of that precision mentioned in the beginning of this blog post. The rest for the code speaks for itself; if not, I'd recommend reading the ConvNet post linked above.

```
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 1
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

### Adding dynamic range quantization

The next step is creating a `TFLiteConverter` which can convert our Keras model into a TFLite representation. Here, we specify that it must optimize the model when doing so, using the `tf.lite.Optimize.DEFAULT` optimization method. In practice, this [reflects](https://www.tensorflow.org/lite/performance/post_training_integer_quant#convert_using_dynamic_range_quantization) dynamic range quantization.

```
# Convert into TFLite model and convert with DEFAULT (dynamic range) quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

Et voila! You can now [save](https://www.reddit.com/r/tensorflow/comments/f1ec02/how_do_i_save_a_converted_tensorflow_lite_model/) your TFLite model, and re-use it on your edge device.

* * *

## Summary

In this article, we looked at quantization for model optimization - in order to make trained machine learning models smaller and faster without incurring performance loss. Quantization involves converting numbers into another number representation, most often from `float32` (TensorFlow default) into `float16` or `int8` formats. This allows models to be smaller, benefiting storage requirements, and often faster, benefiting inference.

We covered multiple forms of quantization: `float16` quantization, where model size is cut in half, as well as full-integer and 16x8-based integer quantization, and finally dynamic range quantization. This involved analyzing how your use case is benefited by one of those approaches, as some work on GPUs, while others work better on CPUs, and so forth. Finally, we covered quantization-aware training, which can be performed prior to quantization, in order to make models robust to loss incurred by quantization by emulating quantization at training time.

I hope this post was useful for your machine learning projects and that you have learned a lot - I definitely did when looking at this topic! In future blogs, we'll cover more optimization aspects from TensorFlow, such as pruning. For now, however, I'd like to point out that if you have any questions or comments, please feel free to leave a comment in the comments section below 💬 Please do the same if you have any comments, remarks or suggestions for improvement; I'll happily adapt the blog to include your feedback.

Thank you for reading MachineCurve today and happy engineering! 😎

\[kerasbox\]

* * *

## References

_Model optimization_. (n.d.). TensorFlow. [https://www.tensorflow.org/lite/performance/model\_optimization](https://www.tensorflow.org/lite/performance/model_optimization)

_Floating-point arithmetic_. (2001, November 11). Wikipedia, the free encyclopedia. Retrieved September 15, 2020, from [https://en.wikipedia.org/wiki/Floating-point\_arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic)

_Why do I have to convert "uint8" into "float32"_. (n.d.). Stack Overflow. [https://stackoverflow.com/questions/59986353/why-do-i-have-to-convert-uint8-into-float32](https://stackoverflow.com/questions/59986353/why-do-i-have-to-convert-uint8-into-float32)

_Here’s why quantization matters for AI_. (2020, August 25). Qualcomm. [https://www.qualcomm.com/news/onq/2019/03/12/heres-why-quantization-matters-ai](https://www.qualcomm.com/news/onq/2019/03/12/heres-why-quantization-matters-ai)

_TensorFlow lite guide_. (n.d.). TensorFlow. [https://www.tensorflow.org/lite/guide](https://www.tensorflow.org/lite/guide)

_Post-training dynamic range quantization_. (n.d.). TensorFlow. [https://www.tensorflow.org/lite/performance/post\_training\_quant](https://www.tensorflow.org/lite/performance/post_training_quant)

_Post-training float16 quantization_. (n.d.). TensorFlow. [https://www.tensorflow.org/lite/performance/post\_training\_float16\_quant](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

_Post-training integer quantization with int16 activations_. (n.d.). TensorFlow. [https://www.tensorflow.org/lite/performance/post\_training\_integer\_quant\_16x8](https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8)

_Post-training integer quantization_. (n.d.). TensorFlow. [https://www.tensorflow.org/lite/performance/post\_training\_integer\_quant](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

_Post-training quantization_. (n.d.). TensorFlow. [https://www.tensorflow.org/lite/performance/post\_training\_quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

_TensorFlow/TensorFlow_. (n.d.). GitHub. [https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize)

_Tfmot.quantization.keras.quantize\_model_. (n.d.). TensorFlow. [https://www.tensorflow.org/model\_optimization/api\_docs/python/tfmot/quantization/keras/quantize\_model](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/quantization/keras/quantize_model)
