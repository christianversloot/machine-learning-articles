---
title: "Tutorial: How to deploy your ConvNet classifier with Keras and FastAPI"
date: "2020-03-19"
categories: 
  - "deep-learning"
  - "frameworks"
tags: 
  - "api"
  - "convolutional-neural-networks"
  - "deep-learning"
  - "deployment"
  - "http"
  - "keras"
  - "machine-learning"
  - "model"
---

Training machine learning models is fun - but what if you found a model that really works? You'd love to deploy it into production, so that others can use it.

In today's blog post, we'll show you how to do this for a ConvNet classifier using Keras and FastAPI. It begins with the software dependencies that we need. This is followed by today's model code, and finally showing you how to run the deployed model.

Are you ready? Let's go! :)

* * *

\[toc\]

* * *

## Software dependencies

In order to complete today's tutorial successfully, and be able to run the model, it's key that you install these software dependencies:

- FastAPI
- Pillow
- Pydantic
- TensorFlow 2.0+
- Numpy

Let's take a look at the dependencies first.

### FastAPI

With FastAPI, we'll be building the _groundwork_ for the machine learning model deployment.

What it is? Simple:

> FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
> 
> FastAPI. (n.d.). [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

With the framework, we can build a web service that accepts requests over HTTP, allows us to receive inputs, and subsequently send the machine learning prediction as the response.

Installing goes through `pip`, with `pip install fastapi`. What's more, you'll also need an ASGI (or Asynchronous Server Gateway Interface) server, such as _uvicorn_: `pip install uvicorn`.

### Pillow

Then Pillow:

> Pillow is the friendly PIL fork by [Alex Clark and Contributors](https://github.com/python-pillow/Pillow/graphs/contributors). PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
> 
> _Pillow — Pillow (PIL Fork) 3.1.2 documentation_. (n.d.). Pillow — Pillow (PIL Fork) 7.0.0 documentation. [https://pillow.readthedocs.io/en/3.1.x/index.html](https://pillow.readthedocs.io/en/3.1.x/index.html)

We can use Pillow to manipulate images - which is what we'll do, as the inputs for our ConvNet are images. Installation, once again, goes through `pip`:

```
pip install Pillow
```

### Pydantic

Now, the fun thing with web APIs is that you can send pretty much anything to them. For example, if you make any call (whether it's a GET one with parameters or a PUT, POST or DELETE one with a body), you can send any data along with your request.

Now, the bad thing with such possibility is that people may send data that is incomprehensible for the machine learning model. For example, it wouldn't work if text was sent instead of an image, or if the image was sent in the wrong way.

Pydantic comes to the rescue here:

> Data validation and settings management using python type annotations.
> 
> Pydantic.[https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)

With this library, we can check whether all data is ok :)

### TensorFlow 2.0+

The need for TensorFlow is obvious - we're deploying a machine learning model.

What's more, we need TensorFlow 2.0+ because of its deep integration with modern Keras, as the model that we'll deploy is a Keras based one.

Fortunately, installing TensorFlow is easy - especially when you're running it on your CPU. [Click here to find out how](https://www.tensorflow.org/install).

### Numpy

Now, last but not least, Numpy. As we all know what it is and what it does, I won't explain it here :) We'll use it for data processing.

* * *

## Today's code

Next up: the code for today's machine learning model deployment 🦾 It consists of three main parts:

- Importing all the necessary libraries.
- Loading the model and getting the input shape.
- Building the FastAPI app.

The latter of which is split into three sub stages:

- Defining the Response.
- Defining the main route.
- Defining the `/prediction` route.

Ready? Let's go! :) Create a Python file, such as `main.py`, on your system, and open it in a code editor. Now, we'll start writing some code :)

### Just a break: what you'll have to do before you go further

Not willing to interrupt, but there are two things that you'll have to do first before you actually build your API:

- Train a machine learning model with Keras, [for example with the MNIST dataset](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/) (we assume that your ML model handles the MNIST dataset from now on, but this doesn't really matter as the API works with all kinds of CNNs).
- Save the model instance, so that you can load it later. [Find out here how](https://www.machinecurve.com/index.php/2020/02/14/how-to-save-and-load-a-model-with-keras/).

### Model imports

The first thing to do is to state all the model imports:

```
# Imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import io
import numpy as np
import sys
```

Obviously, we'll need parts from `FastAPI`, `PIL` (Pillow), `pydantic` and `tensorflow`, as well as `numpy`. But we'll also need a few other things:

- For the list data type, we'll use `typing`
- For input/output operations (specifically, byte I/O), we'll be using `io`
- Finally, we'll need `sys` - for listening to Exception messages.

### Loading the model and getting input shape

Next, we [load the model](https://www.machinecurve.com/index.php/2020/02/14/how-to-save-and-load-a-model-with-keras/):

```
# Load the model
filepath = './saved_model'
model = load_model(filepath, compile = True)
```

This assumes that your model is in the new TensorFlow 2.0 format. If it's not, click the link above, as we describe there how to save it in the 1.0 format - this is directly applicable here.

Then, we get the input shape _as expected by the model_:

```
# Get the input shape for the model layer
input_shape = model.layers[0].input_shape
```

That is, we _wish to know what the model expects_ - so that we can transform any inputs into this shape. We do so by studying the `input_shape` of the first (`i = 0`) layer of our model.

### Building the FastAPI app

Second stage already! Time to build the actual groundwork. First, let's define the FastAPI app:

```
# Define the FastAPI app
app = FastAPI()
```

#### Defining the Response

Then, we can define the Response - or the output that we'll serve if people trigger our web service once it's live. It looks like this:

```
# Define the Response
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[float] = []
  likely_class: int
```

It contains four parts:

- The file name, or `filename`;
- The `contenttype`, or the content type that was found
- A `prediction`, which is a list of floats - [remember how Softmax generates outputs in this way?](https://www.machinecurve.com/index.php/2020/01/08/how-does-the-softmax-activation-function-work/)
- A `likely_class`, which is the most likely class predicted by the model.

#### Defining the main route

Now, we'll define the main route - that is, when people navigate to your web API directly, without going to the `/prediction` route. It's a very simple piece of code:

```
# Define the main route
@app.get('/')
def root_route():
  return { 'error': 'Use POST /prediction instead of the root route!' }
```

It simply tells people to use the correct route.

#### Defining the /prediction route

The `/prediction` route is a slightly longer one:

```
# Define the /prediction route
@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...)):

  # Ensure that this is an image
  if file.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # Convert from RGBA to RGB *to avoid alpha channels*
    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')

    # Convert image into grayscale *if expected*
    if input_shape[3] and input_shape[3] == 1:
      pil_image = pil_image.convert('L')

    # Convert image into numpy format
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    # Scale data (depending on your model)
    numpy_image = numpy_image / 255

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)

    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction.tolist(),
      'likely_class': likely_class
    }
  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))
```

Let's break it into pieces:

- We define the route and the response model, and specify as the parameter that a `File` can be uploaded into the attribute `file`.
- Next, we check the content type of the file - to ensure that it's an image (all image content types start with `image/`, like `image/png`). If it's not, we throw an error - `HTTP 400 Bad Request`.
- Then, we open up a `try/catch` block, where if anything goes wrong the error will be caught gracefully and nicely sent as a Response (`HTTP 500 Internal Server Error`).
- In the `try/catch` block, we first read the contents of the image - into a Byte I/O structure, which acts as a temporary byte storage. We can feed this to `Image` from Pillow, allowing us to actually _open_ the image sent over the network, and manipulate it programmatically.
- Once it's opened, we resize the image so that it meets the `input_shape` of our model.
- Then, we convert the image into `RGB` if it's `RGBA`, to avoid alpha channels (our model hasn't been trained for this).
- If required by the ML model, we convert the image into grayscale.
- Then, we convert it into Numpy format, so that we can manipulate it, and then _scale the image_ (this is dependent on your model! As we scaled [it before training](https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/), we need to do so here too or we get an error)
- Finally, we can generate a prediction and return the Response in the format that we specified.

* * *

## Running the deployed model

That's it already! Now, open up a terminal, navigate to the folder where your `main.py` file is stored, and run `uvicorn main:app --reload` :

```
[32mINFO[0m:     Uvicorn running on [1mhttp://127.0.0.1:8000[0m (Press CTRL+C to quit)
[32mINFO[0m:     Started reloader process [[36m[1m8960[0m]
2020-03-19 20:40:21.560436: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
2020-03-19 20:40:25.858542: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-03-19 20:40:26.763790: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1050 Ti with Max-Q Design major: 6 minor: 1 memoryClockRate(GHz): 1.4175
pciBusID: 0000:01:00.0
2020-03-19 20:40:26.772883: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-03-19 20:40:26.780372: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-03-19 20:40:26.787714: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2020-03-19 20:40:26.797795: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1050 Ti with Max-Q Design major: 6 minor: 1 memoryClockRate(GHz): 1.4175
pciBusID: 0000:01:00.0
2020-03-19 20:40:26.807064: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-03-19 20:40:26.815504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-03-19 20:40:29.059590: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-19 20:40:29.065990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
2020-03-19 20:40:29.071096: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
2020-03-19 20:40:29.076811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2998 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti with Max-Q Design, pci bus id: 0000:01:00.0, compute capability: 6.1)
[32mINFO[0m:     Started server process [[36m19516[0m]
[32mINFO[0m:     Waiting for application startup.
[32mINFO[0m:     Application startup complete.
```

Now, your API has started successfully.

Time to send a request. I'll use Postman for this, which is a HTTP client that is very useful.

I'll send this MNIST sample, as my model was trained on the MNIST dataset:

![](images/image-1.png)

Specifying all the details:

[![](images/image-2-1024x248.png)](https://www.machinecurve.com/wp-content/uploads/2020/03/image-2.png)

Results in this output:

```
{
    "filename": "mnist_sample.png",
    "contenttype": "image/png",
    "prediction": [
        0.0004434768052306026,
        0.003073320258408785,
        0.008758937008678913,
        0.0034302924759685993,
        0.0006626666290685534,
        0.0021806098520755768,
        0.000005191866875975393,
        0.9642654657363892,
        0.003465399844571948,
        0.013714754022657871
    ],
    "likely_class": 7
}
```

Oh yeah! 🎉

* * *

## Summary

In this blog post, we've seen how machine learning models can be deployed by means of a web based API. I hope you've learnt something today. If you did, please leave a comment in the comments section! :)

Sorry for the long delay in blogs again and happy engineering. See you soon! 😎

### Full model code

If you wish to obtain the code at once, here you go:

```
# Imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import io
import numpy as np
import sys

# Load the model
filepath = './saved_model'
model = load_model(filepath, compile = True)

# Get the input shape for the model layer
input_shape = model.layers[0].input_shape

# Define the FastAPI app
app = FastAPI()

# Define the Response
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[float] = []
  likely_class: int

# Define the main route
@app.get('/')
def root_route():
  return { 'error': 'Use GET /prediction instead of the root route!' }

# Define the /prediction route
@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...)):

  # Ensure that this is an image
  if file.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # Convert from RGBA to RGB *to avoid alpha channels*
    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')

    # Convert image into grayscale *if expected*
    if input_shape[3] and input_shape[3] == 1:
      pil_image = pil_image.convert('L')

    # Convert image into numpy format
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    # Scale data (depending on your model)
    numpy_image = numpy_image / 255

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)

    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction.tolist(),
      'likely_class': likely_class
    }
  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))
```

\[kerasbox\]

* * *

## References

FastAPI. (n.d.). [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

_Pillow — Pillow (PIL Fork) 3.1.2 documentation_. (n.d.). Pillow — Pillow (PIL Fork) 7.0.0 documentation. [https://pillow.readthedocs.io/en/3.1.x/index.html](https://pillow.readthedocs.io/en/3.1.x/index.html)

Pydantic.[https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)
