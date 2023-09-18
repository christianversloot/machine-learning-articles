---
title: "Commoditizing AI? The state of automated machine learning."
date: "2019-07-22"
categories:
  - "deep-learning"
  - "svms"
tags:
  - "automl"
  - "commoditization"
  - "deep-learning"
  - "machine-learning"
---

It cannot go unnoticed that machine learning has been quite the hype these past few years. AI programs at universities are spiking in interest and, at least here in the Netherlands, students have to be told not to come because they are so crowded.

However, hidden from popular belief, are we on the verge of a radical transformation in machine learning and its subset practice of deep learning?

A transformation in the sense that we are moving towards automated machine learning - making hardcore ML jobs obsolete?

Perhaps so, as recent research reports indicate that research into automated ML tools is intensifying (Tuggener et al., 2019). It triggered me: can I lose my job as a ML engineer even _before_ the field has stopped to be hot?

Let's find out. In this blog, I'll take a brief look into so-called _AutoML_ tools as well as their developments. I first take a theoretical path and list the main areas of research into automating ML. I'll then identify a few practical tools that I think are most promising today. Finally, I'll discuss how this may in my opinion affect our jobs as ML engineers.

\[ad\]

\[toc\]

\[ad\]

## What are the reasons for automating machine learning?

Data scientists have the sexiest job of the 21st Century, at least that's what they wrote some years back. However, the job is really complex, especially when it comes to training machine learning models. It encompasses many things...

The first step is getting to know your data. What are its ideosyncrasies? What is important in the dataset? Which features do you think are most discriminative with respect to the machine learning problem at hand? Those are questions that must be answered by data scientists before one can even think about training a ML model.

Then, next question - which type of model must be used? Should we use Support Vector Machines with some kernel function that allows us to train SVMs for non-linear datasets? Or should we use neural networks instead? If so, what type of neural network?

[![](images/confused-digital-nomad-electronics-874242-1024x682.jpg)](https://machinecurve.com/wp-content/uploads/2019/07/confused-digital-nomad-electronics-874242-1024x682.jpg)

How ML engineers may feel every now and then.

Ok, suppose that we chose a certain class of neural networks, say [Convolutional Neural Networks](https://machinecurve.com/index.php/2018/12/07/convolutional-neural-networks-and-their-components-for-computer-vision/). You'll then have to decide about the network architecture. How many layers should be used? Which activation functions must be added to these layers? What kind of regularization do I apply? How many densely-classified layer must accompany my convolutional ones? All kind of questions that must be answered by the engineer.

Suppose that you have chosen both a _model class_ and an _architecture_. You'll then move on and select a set of hyperparameters, or configuration options. Example ones are the degree with which a model is optimized every iteration, also known as the learning rate. Similarly, you choose the optimizer, and the loss function to be used during training. And there are other ones.

All right, but how do I even start when I already feel overwhelmed right now?

### Data science success correlates with experience

Quite easy. Very likely, you do not know the answers to all these questions in advance. Often, you therefore use the experience you have to guide you towards an intuitively suitable algorithm. Subsequently, you experiment with various architectures and hyperparameters - slightly guided by what is found in the literature, perhaps, but often based on common sense.

And worry not: it's not strange that difficult jobs are made easier. In fact, this is very common. In the 1990s and later, the World Wide Web caused a large increase in access to information. This made difficult jobs, such as collecting insights on highly specific topics, much easier and - often - obsolete. This process can now also be observed in the field of machine learning.

Will AI become a commodity? Let's see where we stand now, both in theory and in practice.

\[ad\]

## What are current approaches towards automated ML?

What becomes clear from the paper written by [Tuggener et al. (2019)](https://arxiv.org/abs/1907.08392) is that much research is currently being performed into automating "various blocks of the machine learning pipeline", i.e. from the beginning to the end. They suggest that these developments can be grouped into these distinct categories:

- Automating feature engineering.
- Meta-learning.
- Architecture search.
- Hyperparameter optimization.
- Combined Model Selection and Hyperparameter Optimization (CASH).

### Automating feature engineering

The first category is automated **feature engineering**. Every model harnesses feature vectors and, together with their respective targets in the case of supervised learning, attempts to identify patterns in the data set.

However, not every feature in a feature vector is, so to say, _discriminative_ enough.

That is, it blinds the model from identifying relevant patterns rather than making those patterns clearer.

It's often best to remove these features. This is often a tedious job, since an engineer must predict which ones must be removed, before retraining the models to see whether his or her prediction is right.

Various approaches towards automating this problem exist today. For example, it can be considered to be a reinforcement learning problem, where an intelligent agent learns to recognize good and bad features. Other techniques combine features before feeding them into the model, assessing their effectiveness. Another approach attempts to compute the information gain for scenarios where features are varied. Their goal is to maximize this gain. However, they all have in common that they _only focus on the feature engineering aspects_. That's however only one aspect of the ML pipeline.

### Meta-learning

In another approach, named **meta-learning**, the features are not altered. Rather, a meta-model is trained that has learnt from previous training processes. Such models can take as input e.g. the number and type of features as well as the algorithms and then generate a prediction with respect to what optimization is necessary.

As Tuggener et al. (2019) demonstrate, many such algorithms are under active development today. The same observation is made by Elshawi et al. (2019).

### Architecture search

Similarly under active research scrutiny these days is what Tuggener et al. (2019) call **architecture search**. In essence, finding the best-performing model can be considered to be a search problem with the goal of finding the right model architecture. It's therefore perhaps one of the most widely used means for automating ML these days.

Within this category, many sub approaches to searching the most optimal architecture can be observed today (Elshawi et al., 2019). At a very high level, they are as follows:

- Searching randomly. It's a naïve approach, but apparently especially this fact benefits finding model architectures.
- Reinforcement learning, or training a dumb agent by means of "losses" and "rewards" to recognize good paths towards improvement, is an approach that is used today as well.
- By optimizing the gradient of the _search problem_, one can essentially consider finding the architecture to be a meta problem.
- Evolutionary algorithms that add genetic optimization can be used for finding well-performing architectures.
- Bayesian optimization, or selecting a path to improvement from a Gaussian distribution, is also used in certain works.

I refer to the original work (Elshawi et al., 2019 - see the references list below) for a more detailed review.

### Hyperparameter optimization

Suppose that you have chosen a particular model type, say a Convolutional Neural Network. As you've read before, you then face the choice of hyperparameter selection - or, selecting the model's configuration elements.

It includes, as we recall, picking a suitable optimizer, setting the learning rate, et cetera.

This is essentially a large search problem to be solved.

If **hyperparameter optimization** is used for automating machine learning, it's essentially this last part of the ML training process that is optimized.

But is it enough? Let's introduce CASH.

### Combined Model Selection and Hyperparameter Optimization (CASH)

If you combine the approaches discussed previously, you come to what is known as the CASH approach: combining model selection and hyperparameter optimization.

Suppose that you have a dataset for which you wish to train a machine learning model, but you haven't decided yet about an architecture.

Solving the CASH problem would essentially mean that you find the optimum data pipeline for the dataset (Tuggener et al., 2019) - including:

- Cleaning your data.
- Feature selection and construction, where necessary.
- Model selection (SVMs? Neural networks? CNNs? RNNs? Eh, who knows?)
- Hyperparameter optimization.
- Perhaps, even ensemble learning, combining the models into a better-performing ensemble.

According to Tuggener et al. (2019) this would save a massive amount of time for data scientists. They argued that a problem which their data scientists worked hard on for weeks could be solved by automated tooling in 30 minutes. Man, that's progress.

\[ad\]

## AutoML tools used in practice

All these theoretical contributions are nice, but I am way more curious about how they are applied in practice.

What systems for automating machine learning are in use today?

Let's see if we can find some and compare them.

### Cloud AutoML

The first system I found is called [Cloud AutoML](https://cloud.google.com/automl/) and is provided as a service by Google. It suggests that it uses Google's _Neural Architecture Search_. This yields the insight that it therefore specifically targets neural networks and attempts to find the best architecture with respect to the dataset. It focuses on computer vision, natural language processing and tabular data.

[![](images/art-blue-skies-clouds-335907-1024x686.jpg)](https://machinecurve.com/wp-content/uploads/2019/07/art-blue-skies-clouds-335907.jpg)

The cloud is often the place for automated machine learning, but this does not always have to be the case.

### AutoKeras

Cloud AutoML is however rather pricy as it apparently costs $20 per hour (Seif, 2019). Fortunately, for those who have experience with [Keras](https://machinecurve.com/index.php/mastering-keras/), there is now a library out there called AutoKeras - take a look at it [here](https://github.com/keras-team/autokeras). It essentially turns the Keras based way of working into an AutoML problem: it performs an architecture search by means of Bayesian optimization and network morphism. Back to plain English now, but if you really wish to understand it deeper - take a look at (Jin et al., 2018).

I do - and will dive deeper into it ASAP. Remind me of this, please! 😄

### Other tools

A post by Oleksii Kharkovyna at Medium/TowardsDataScience suggests that there are various other approaches to automated ML in use today. Check it out [here](https://towardsdatascience.com/top-10-data-science-ml-tools-for-non-programmers-d12ce6dcccc).

## My conclusions

The field of machine learning seems to be democratizing rapidly. Whereas deep knowledge on algorithms, particularly deep neural networks these days, was required in the past, that seems to be less and less the case.

Does this mean that no ML engineers are required anymore?

No. Not in my view.

![](images/connection-data-desk-1181675-1024x683.jpg)

However, what I'm trying to suggest here is a number of things:

1. Do not stare yourself blind at becoming an expert in model optimization. It's essentially a large search problem that is bound to be democratized and, by consequently, automated away.
2. Take notice of the wide array of automated machine learning tools and get experience with them. You may be asked to use them in the future. It would be nice if you already had some experience - it would set you apart from the rest 😄
3. Become creative! 🧠 These automated machine learning solutions are simply the solvers of large search problems. However, translating business problems into a machine learning task still requires creativity and tactical and/or strategic awareness. This is still a bridge too far for those kind of technologies.

Data science may still be the sexiest job of the 21st Century, but be prepared for some change. Would you agree with me? Or do you disagree entirely? I would be glad to know. Leave your comments in the comment section below 👇 I'll respond with my thoughts as soon as I can.

\[ad\]

## References

Elshawi, R., Maher, M., & Sakr, S. (2019, June). Automated Machine Learning: State-of-The-Art and Open Challenges. Retrieved from [https://arxiv.org/abs/1906.02287](https://arxiv.org/abs/1906.02287)

Kharkovyna, O. (2019, May 22). Top 10 Data Science & ML Tools for Non-Programmers - Towards Data Science. Retrieved from [https://towardsdatascience.com/top-10-data-science-ml-tools-for-non-programmers-d12ce6dcccc](https://towardsdatascience.com/top-10-data-science-ml-tools-for-non-programmers-d12ce6dcccc ﻿)

Jin, H., Song, Q., & Hu, X. (2018, June). Auto-Keras: An Efficient Neural Architecture Search System. Retrieved from [https://arxiv.org/abs/1806.10282](https://arxiv.org/abs/1806.10282)

Seif, G. (2019, February 23). AutoKeras: The Killer of Google's AutoML - Towards Data Science. Retrieved from [https://towardsdatascience.com/autokeras-the-killer-of-googles-automl-9e84c552a319](https://towardsdatascience.com/autokeras-the-killer-of-googles-automl-9e84c552a319)

Tuggener, L., Amirian, M., Rombach, K., Lörwald, S., Varlet, A., Westermann, C., & Stadelmann, T. (2019, July). Automated Machine Learning in Practice: State of the Art and Recent Results. Retrieved from [https://arxiv.org/abs/1907.08392](https://arxiv.org/abs/1907.08392)
