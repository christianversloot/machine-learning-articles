---
title: "Transformers for Long Text: Code Examples with Longformer"
date: "2021-03-12"
categories: 
  - "buffer"
  - "deep-learning"
  - "frameworks"
tags: 
  - "code-examples"
  - "deep-learning"
  - "longformer"
  - "machine-learning"
  - "transformer"
---

Transformer models have been boosting NLP for a few years now. Every now and then, new additions make them even more performant. Longformer is one such extension, as it can be used for long texts.

While being applied for many tasks - think [machine translation](https://www.machinecurve.com/index.php/2021/02/16/easy-machine-translation-with-machine-learning-and-huggingface-transformers/), [text summarization](https://www.machinecurve.com/index.php/2020/12/21/easy-text-summarization-with-huggingface-transformers-and-machine-learning/) and [named-entity recognition](https://www.machinecurve.com/index.php/2021/02/11/easy-named-entity-recognition-with-machine-learning-and-huggingface-transformers/) - classic Transformers always have faced difficulties when texts became too long. This results from the self-attention mechanism applied in these models, which in terms of time and memory consumption scales quadratically with sequence length.

Longformer makes Transformers available to long texts by introducing a sparse attention mechanism and combining it with a global, task specific one. More about that [can be read here](https://www.machinecurve.com/index.php/question/what-is-the-longformer-transformer-and-how-does-it-work/). In this tutorial, you're going to work with actual Longformer instances, for a variety of tasks. More specifically, after reading it, you will know...

- **How to use Longformer based Transformers in your Machine Learning project.**
- **What is necessary for using Longformer for Question Answering, Text Summarization and Masked Language Modeling (Missing Text Prediction).**
- **That Longformer is really capable of handling large texts, as we demonstrate in our examples.**

Let's take a look! 🚀

* * *

\[toc\]

* * *

## What is the Longformer model?

Ever since Transformer models have been introduced in 2017, they have brought about change in the world of NLP. With a variety of architectures, such as [BERT](https://www.machinecurve.com/index.php/2021/01/04/intuitive-introduction-to-bert/) and [GPT](https://www.machinecurve.com/index.php/2021/01/02/intuitive-introduction-to-openai-gpt/), a wide range of language tasks have been improved to sometimes human-level quality... and in addition, with libraries like HuggingFace Transformers, applying them has been democratized significantly.

As a consequence, we can now create pipelines for [machine translation](https://www.machinecurve.com/index.php/2021/02/16/easy-machine-translation-with-machine-learning-and-huggingface-transformers/), [text summarization](https://www.machinecurve.com/index.php/2020/12/21/easy-text-summarization-with-huggingface-transformers-and-machine-learning/) and [named-entity recognition](https://www.machinecurve.com/index.php/2021/02/11/easy-named-entity-recognition-with-machine-learning-and-huggingface-transformers/) with only a few lines of code.

Classic Transformers - including GPT and BERT - have one problem though: the time and memory complexity of the [self-attention function](https://www.machinecurve.com/index.php/2020/12/28/introduction-to-transformers-in-machine-learning/#multi-head-attention). As you may recall, this function applies queries, keys and values by means of \[latex\]Q\[/latex\], \[latex\]K\[/latex\] and \[latex\]V\[/latex\] generations from the input embeddings - and more specifically, it performs a multiplication of the sort \[latex\]QK^T\[/latex\]. This multiplication is _quadratic_. In other words, time and memory complexity increases quadratically with sequence length.

In other words, when your sequences (and thus your input length) are really long, Transformers cannot process them anymore - simply because too much time or too many resources are required. To mitigate this, classic Transformers and BERT- and GPT-like approaches truncate text and sometimes adapt their architecture to specific tasks.

While we want a Transformer that can handle long texts without the necessity for any significant changes.

That's why **Longformer** was introduced. It changes the attention mechanism by applying _dilated sliding window based attention_, where each token has a 'window' of tokens around that particular token - including dilation - for which attention is computed. In other words, attention is now more _local_ rather than global. To ensure that some global patterns are captured as well (e.g. specific attention to particular tokens), _global attention_ is added as well - but this is more task specific. We have covered the details of Longformer [in another article](https://www.machinecurve.com/index.php/question/what-is-the-longformer-transformer-and-how-does-it-work/), so make sure to head there if you want to understand Longformer in more detail. Let's now take a look at the example text that we will use today, and then move forward to the code examples.

### Today's example text

To show you that Longformer works with really long tasks in a variety of tasks, we'll use some segments from the [Wikipedia page about Germany](https://en.wikipedia.org/wiki/Germany) (Wikipedia, 2001). More specifically, we will be using this text:

```
Germany (German: Deutschland, German pronunciation: [ˈdɔʏtʃlant]), officially the Federal Republic of Germany,[e] is a country at the intersection of Central and Western Europe. It is situated between the Baltic and North seas to the north, and the Alps to the south; covering an area of 357,022 square kilometres (137,847 sq mi), with a population of over 83 million within its 16 constituent states. It borders Denmark to the north, Poland and the Czech Republic to the east, Austria and Switzerland to the south, and France, Luxembourg, Belgium, and the Netherlands to the west. Germany is the second-most populous country in Europe after Russia, as well as the most populous member state of the European Union. Its capital and largest city is Berlin, and its financial centre is Frankfurt; the largest urban area is the Ruhr.

Various Germanic tribes have inhabited the northern parts of modern Germany since classical antiquity. A region named Germania was documented before AD 100. In the 10th century, German territories formed a central part of the Holy Roman Empire. During the 16th century, northern German regions became the centre of the Protestant Reformation. Following the Napoleonic Wars and the dissolution of the Holy Roman Empire in 1806, the German Confederation was formed in 1815. In 1871, Germany became a nation-state when most of the German states unified into the Prussian-dominated German Empire. After World War I and the German Revolution of 1918–1919, the Empire was replaced by the semi-presidential Weimar Republic. The Nazi seizure of power in 1933 led to the establishment of a dictatorship, World War II, and the Holocaust. After the end of World War II in Europe and a period of Allied occupation, Germany was divided into the Federal Republic of Germany, generally known as West Germany, and the German Democratic Republic, East Germany. The Federal Republic of Germany was a founding member of the European Economic Community and the European Union, while the German Democratic Republic was a communist Eastern Bloc state and member of the Warsaw Pact. After the fall of communism, German reunification saw the former East German states join the Federal Republic of Germany on 3 October 1990—becoming a federal parliamentary republic led by a chancellor.

Germany is a great power with a strong economy; it has the largest economy in Europe, the world's fourth-largest economy by nominal GDP, and the fifth-largest by PPP. As a global leader in several industrial, scientific and technological sectors, it is both the world's third-largest exporter and importer of goods. As a developed country, which ranks very high on the Human Development Index, it offers social security and a universal health care system, environmental protections, and a tuition-free university education. Germany is also a member of the United Nations, NATO, the G7, the G20, and the OECD. It also has the fourth-greatest number of UNESCO World Heritage Sites.
```

### Software requirements

To run the code that you will create in the next sections, it is important that you have installed a few things here and there. Make sure to have an environment (preferably) or a global Python environment running on your machine. Then make sure that HuggingFace Transformers is installed through `pip install transformers`. As HuggingFace Transformers runs on top of either PyTorch or TensorFlow, install any of the two.

Note that the code examples below are built for PyTorch based HuggingFace. They can be adapted to TensorFlow relatively easily, usually by prepending `TF` before the model you are importing, e.g. `TFAutoModel`.

### Moving forward

Now, we can move forward to showing you how to use Longformer. Specifically, you're going to see code for these tasks:

- Question Answering
- Text Summarization
- Masked Language Modeling (Predicting missing text).

Let's take a look! 🚀

* * *

## Longformer and Question Answering

Longformer can be used for question answering tasks. This requires that the pretrained Longformer is fine-tuned so that it is tailored to the task. Today, you're going to use a Longformer model that has been fine-tuned on the [SQuAD v1](https://www.machinecurve.com/index.php/question/what-is-the-squad-dataset/) language task.

This is a question answering task using the Stanford Question Answering Dataset (SQuAD).

Creating the code involves the following steps:

1. **Imports:** we'll need PyTorch itself to take an `argmax` with gradients later, so we must import it through `import torch`. Then, we also need the `AutoTokenizer` and the `AutoModelForQuestionAnswering` from HuggingFace `transformers`.
2. **Initialization of tokenizer and model.** Secondly, we need to get our tokenizer and model up and running. For doing so, we'll be using a model that is available in the HuggingFace Model Hub - the `valhalla/longformer-base-4096-finetuned-squadv1` model. As you can see, it's the Longformer base model fine-tuned on SQuAD v1. As with [any fine-tuned Longformer model](https://www.machinecurve.com/index.php/question/what-is-the-longformer-transformer-and-how-does-it-work/), it can support up to 4096 tokens in a sequence.
3. **Specifying the text and the question**. The `text` contains the context that is used by Longformer for answering the question. As you can imagine, it's the text that we specified above. For the `question`, we're interested in the size of Germany's economy by national GDP (Germany has the fourth-largest economy can be read in the text).
4. **Tokenization of the input text**. Before we can feed the text to our Longformer model, we must tokenize it. We simply feed question and text to the tokenizer and return PyTorch tensors. From these, we can extract the input identifiers, i.e. the unique token identifiers in the vocabulary for the tokens from our input text.
5. **Getting the attention mask**. Recall that Longformer works with sparse local attention and task-specific global attention. For question answering, the tokenizer generates the attention mask; this was how the tokenizer was trained. That's why we can also extract the attention mask from the encoding. Note that global attention is applied to tokens related to the question only.
6. **Getting the predictions**. Once we have tokenized our input and retrieved the atetntion mask, we can get the predictions.
7. **Converting the predictions into the answer, and printing the answer on screen.** The seventh and final step is to actually convert the identifiers to tokens, which we then decode and print on our screen.

```
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

# Initialize the model
model = AutoModelForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

# Specify text and question
text = """Germany (German: Deutschland, German pronunciation: [ˈdɔʏtʃlant]), officially the Federal Republic of Germany,[e] is a country at the intersection of Central and Western Europe. It is situated between the Baltic and North seas to the north, and the Alps to the south; covering an area of 357,022 square kilometres (137,847 sq mi), with a population of over 83 million within its 16 constituent states. It borders Denmark to the north, Poland and the Czech Republic to the east, Austria and Switzerland to the south, and France, Luxembourg, Belgium, and the Netherlands to the west. Germany is the second-most populous country in Europe after Russia, as well as the most populous member state of the European Union. Its capital and largest city is Berlin, and its financial centre is Frankfurt; the largest urban area is the Ruhr.Various Germanic tribes have inhabited the northern parts of modern Germany since classical antiquity. A region named Germania was documented before AD 100. In the 10th century, German territories formed a central part of the Holy Roman Empire. During the 16th century, northern German regions became the centre of the Protestant Reformation. Following the Napoleonic Wars and the dissolution of the Holy Roman Empire in 1806, the German Confederation was formed in 1815. In 1871, Germany became a nation-state when most of the German states unified into the Prussian-dominated German Empire. After World War I and the German Revolution of 1918–1919, the Empire was replaced by the semi-presidential Weimar Republic. The Nazi seizure of power in 1933 led to the establishment of a dictatorship, World War II, and the Holocaust. After the end of World War II in Europe and a period of Allied occupation, Germany was divided into the Federal Republic of Germany, generally known as West Germany, and the German Democratic Republic, East Germany. The Federal Republic of Germany was a founding member of the European Economic Community and the European Union, while the German Democratic Republic was a communist Eastern Bloc state and member of the Warsaw Pact. After the fall of communism, German reunification saw the former East German states join the Federal Republic of Germany on 3 October 1990—becoming a federal parliamentary republic led by a chancellor.Germany is a great power with a strong economy; it has the largest economy in Europe, the world's fourth-largest economy by nominal GDP, and the fifth-largest by PPP. As a global leader in several industrial, scientific and technological sectors, it is both the world's third-largest exporter and importer of goods. As a developed country, which ranks very high on the Human Development Index, it offers social security and a universal health care system, environmental protections, and a tuition-free university education. Germany is also a member of the United Nations, NATO, the G7, the G20, and the OECD. It also has the fourth-greatest number of UNESCO World Heritage Sites."""
question = "How large is Germany's economy by nominal GDP?"

# Tokenize the input text
encoding = tokenizer(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]

# Get attention mask (local + global attention)
attention_mask = encoding["attention_mask"]

# Get the predictions
start_scores, end_scores = model(input_ids, attention_mask=attention_mask).values()

# Convert predictions into answer
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

# Print answer
print(answer)
```

The results:

```
fourth-largest
```

Yep indeed, Germany has the fourth-largest economy by nominal GDP. Great! :D

* * *

## Longformer and Text Summarization

Next up is text summarization. This can also be done with Transformers. Compared to other tasks such as question answering, summarization is a _generative_ activity that also greatly benefits from a lot of _context_. That's why traditionally, sequence-to-sequence architectures have been useful for this purpose.

That's why in the example below, we are using a Longformer2RoBERTa architecture, which utilizes Longformer as the encoder segment, and RoBERTa as the decoder segment. It was fine-tuned on the CNN/DailyMail dataset, which is a common one in the field of text summarization.

So strictly speaking, this is not a full Longformer model, but Longformer merely plays a part in the whole stack. Nevertheless, it works pretty well, as we shall see!

This is how we build the model

- **Imports:** we import the `LongformerTokenizer` and the `EncoderDecoderModel` (which is what you'll need for [Seq2Seq](https://www.machinecurve.com/index.php/2020/12/29/differences-between-autoregressive-autoencoding-and-sequence-to-sequence-models-in-machine-learning/)!)
- **Loading model and tokenizer:** our model is an instance of `patrickvonplaten/longformer2roberta-cnn_dailymail-fp16`, which contains the full Seq2Seq model. However, as the encoder segment is Longformer, we can use the Longformer tokenizer - so we use `allenai/longformer-base-4096` there.
- **Specifying the article:** the text from above.
- **Tokenization, summarization and conversion:** we feed the `article` into the tokenizer, return the input ids from the PyTorch based Tensors, and then generate the summary with our `model`. Once the summary is there, we use the `tokenizer` again for decoding the output identifiers into readable text. We skip special tokens.
- **Printing the summary on screen:** to see if it works :)

```
from transformers import LongformerTokenizer, EncoderDecoderModel

# Load model and tokenizer
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 

# Specify the article
article = """Germany (German: Deutschland, German pronunciation: [ˈdɔʏtʃlant]), officially the Federal Republic of Germany,[e] is a country at the intersection of Central and Western Europe. It is situated between the Baltic and North seas to the north, and the Alps to the south; covering an area of 357,022 square kilometres (137,847 sq mi), with a population of over 83 million within its 16 constituent states. It borders Denmark to the north, Poland and the Czech Republic to the east, Austria and Switzerland to the south, and France, Luxembourg, Belgium, and the Netherlands to the west. Germany is the second-most populous country in Europe after Russia, as well as the most populous member state of the European Union. Its capital and largest city is Berlin, and its financial centre is Frankfurt; the largest urban area is the Ruhr.Various Germanic tribes have inhabited the northern parts of modern Germany since classical antiquity. A region named Germania was documented before AD 100. In the 10th century, German territories formed a central part of the Holy Roman Empire. During the 16th century, northern German regions became the centre of the Protestant Reformation. Following the Napoleonic Wars and the dissolution of the Holy Roman Empire in 1806, the German Confederation was formed in 1815. In 1871, Germany became a nation-state when most of the German states unified into the Prussian-dominated German Empire. After World War I and the German Revolution of 1918–1919, the Empire was replaced by the semi-presidential Weimar Republic. The Nazi seizure of power in 1933 led to the establishment of a dictatorship, World War II, and the Holocaust. After the end of World War II in Europe and a period of Allied occupation, Germany was divided into the Federal Republic of Germany, generally known as West Germany, and the German Democratic Republic, East Germany. The Federal Republic of Germany was a founding member of the European Economic Community and the European Union, while the German Democratic Republic was a communist Eastern Bloc state and member of the Warsaw Pact. After the fall of communism, German reunification saw the former East German states join the Federal Republic of Germany on 3 October 1990—becoming a federal parliamentary republic led by a chancellor.Germany is a great power with a strong economy; it has the largest economy in Europe, the world's fourth-largest economy by nominal GDP, and the fifth-largest by PPP. As a global leader in several industrial, scientific and technological sectors, it is both the world's third-largest exporter and importer of goods. As a developed country, which ranks very high on the Human Development Index, it offers social security and a universal health care system, environmental protections, and a tuition-free university education. Germany is also a member of the United Nations, NATO, the G7, the G20, and the OECD. It also has the fourth-greatest number of UNESCO World Heritage Sites."""

# Tokenize and summarize
input_ids = tokenizer(article, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)

# Get the summary from the output tokens
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print summary
print(summary)
```

The results:

```
Germany is the second-most populous country in Europe after Russia.
It is the country's second-largest economy and the most populous member state of the European Union.
Germany is also a member of the United Nations, the G7, the OECD and the G20.
```

Quite a good summary indeed!

* * *

## Longformer and Masked Language Modeling / Predicting Missing Text

Next up is Masked Language Modeling using Longformer. Recall that [MLM](https://www.machinecurve.com/index.php/2021/03/02/easy-masked-language-modeling-with-machine-learning-and-huggingface-transformers/) is a technique used for pretraining BERT-style models. When applied, parts of the text are masked, and the goal of the model is to predict the original text. If it can do so correctly and at scale, it effectively learns the relationships between text and therefore generates the supervision signal through the attention mechanism.

Let's see if we can get this to work with Longformer, so that we can apply MLM to longer texts. As you can see we apply the mask just after the text starts: `officially the Federal Republic of Germany,[e] is a {mask}`.

That should be _country_, indeed, so let's see if we can get the model to produce that.

1. **Imports and pipeline init:** HuggingFace Transformers offers a [`pipeline` for Masked Language Modeling](https://www.machinecurve.com/index.php/2021/03/02/easy-masked-language-modeling-with-machine-learning-and-huggingface-transformers/), the `fill-mask` pipeline. We can initialize it with the `allenai/longformer-base-4096` model. This base model is the MLM pretrained base model that still requires fine-tuning for task specific behavior. However, because it was pretrained with MLM, we can also _use_ it for MLM and thus Predicting Missing Text. We thus load the `pipeline` API from `transformers`.
2. **Loading the mask token:** the `mlm.tokenizer` has a specific `mask_token`. We simplify it by referring to it as `mask`.
3. **Masking the text:** we specify the text, but then apply `{mask}` to where `country` is written in the original text.
4. **Perform MLM:** we then feed the `text` to our `mlm` pipeline to obtain the result, which we then print on screen.

```
from transformers import pipeline

# Initialize MLM pipeline
mlm = pipeline('fill-mask', model='allenai/longformer-base-4096')

# Get mask token
mask = mlm.tokenizer.mask_token

# Get result for particular masked phrase
text = f"""Germany (German: Deutschland, German pronunciation: [ˈdɔʏtʃlant]), officially the Federal Republic of Germany,[e] is a {mask} at the intersection of Central and Western Europe. It is situated between the Baltic and North seas to the north, and the Alps to the south; covering an area of 357,022 square kilometres (137,847 sq mi), with a population of over 83 million within its 16 constituent states. It borders Denmark to the north, Poland and the Czech Republic to the east, Austria and Switzerland to the south, and France, Luxembourg, Belgium, and the Netherlands to the west. Germany is the second-most populous country in Europe after Russia, as well as the most populous member state of the European Union. Its capital and largest city is Berlin, and its financial centre is Frankfurt; the largest urban area is the Ruhr.Various Germanic tribes have inhabited the northern parts of modern Germany since classical antiquity. A region named Germania was documented before AD 100. In the 10th century, German territories formed a central part of the Holy Roman Empire. During the 16th century, northern German regions became the centre of the Protestant Reformation. Following the Napoleonic Wars and the dissolution of the Holy Roman Empire in 1806, the German Confederation was formed in 1815. In 1871, Germany became a nation-state when most of the German states unified into the Prussian-dominated German Empire. After World War I and the German Revolution of 1918–1919, the Empire was replaced by the semi-presidential Weimar Republic. The Nazi seizure of power in 1933 led to the establishment of a dictatorship, World War II, and the Holocaust. After the end of World War II in Europe and a period of Allied occupation, Germany was divided into the Federal Republic of Germany, generally known as West Germany, and the German Democratic Republic, East Germany. The Federal Republic of Germany was a founding member of the European Economic Community and the European Union, while the German Democratic Republic was a communist Eastern Bloc state and member of the Warsaw Pact. After the fall of communism, German reunification saw the former East German states join the Federal Republic of Germany on 3 October 1990—becoming a federal parliamentary republic led by a chancellor.Germany is a great power with a strong economy; it has the largest economy in Europe, the world's fourth-largest economy by nominal GDP, and the fifth-largest by PPP. As a global leader in several industrial, scientific and technological sectors, it is both the world's third-largest exporter and importer of goods. As a developed country, which ranks very high on the Human Development Index, it offers social security and a universal health care system, environmental protections, and a tuition-free university education. Germany is also a member of the United Nations, NATO, the G7, the G20, and the OECD. It also has the fourth-greatest number of UNESCO World Heritage Sites."""
result = mlm(text)

# Print result
print(result)
```

When we observe the results (we cut off the text at the masked token; it continues in the real results), we can see that it is capable of predicting `country` indeed!

```
[{'sequence': "Germany (German: Deutschland, German pronunciation: [ˈdɔʏtʃlant]), officially the Federal Republic of Germany,[e] is a country
```

Great!

* * *

## Summary

In this tutorial, we covered practical aspects of the Longformer Transformer model. Using this model, you can now process really long texts, by means of the simple change in attention mechanism compared to the one used in classic Transformers. Put briefly, you have learned...

- **How to use Longformer based Transformers in your Machine Learning project.**
- **What is necessary for using Longformer for Question Answering, Text Summarization and Masked Language Modeling (Missing Text Prediction).**
- **That Longformer is really capable of handling large texts, as we demonstrate in our examples.**

I hope that this article was useful to you! If it was, please let me know through the comments 💬 Please do the same if you have any questions or other comments. I'd love to hear from you :)

Thank you for reading MachineCurve today and happy engineering! 😎

* * *

## References

HuggingFace. (n.d.). _Allenai/longformer-base-4096 · Hugging face_. Hugging Face – On a mission to solve NLP, one commit at a time. [https://huggingface.co/allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096)

HuggingFace. (n.d.). _Valhalla/longformer-base-4096-finetuned-squadv1 · Hugging face_. Hugging Face – On a mission to solve NLP, one commit at a time. [https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1](https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1)

HuggingFace. (n.d.). _Patrickvonplaten/longformer2roberta-cnn\_dailymail-fp16 · Hugging face_. Hugging Face – On a mission to solve NLP, one commit at a time. [https://huggingface.co/patrickvonplaten/longformer2roberta-cnn\_dailymail-fp16](https://huggingface.co/patrickvonplaten/longformer2roberta-cnn_dailymail-fp16)

Wikipedia. (2001, November 9). _Germany_. Wikipedia, the free encyclopedia. Retrieved March 12, 2021, from [https://en.wikipedia.org/wiki/Germany](https://en.wikipedia.org/wiki/Germany)
