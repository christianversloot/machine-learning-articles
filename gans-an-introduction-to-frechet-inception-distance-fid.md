---
title: "GANs: an Introduction to Fréchet Inception Distance (FID)"
date: "2021-11-09"
categories:
  - "deep-learning"
tags:
  - "deep-learning"
  - "fid"
  - "frechet-inception-distance"
  - "machine-learning"
  - "neural-networks"
---

The **Frechet Inception Distance** or FID is a method for comparing the statistics of two distributions by computing the distance between them. In GANs, the FID method is used for computing how much the distribution of the Generator looks like the distribution of the Discriminator. By consequence, it is a metric of GAN performance – the lower the FID, the better the GAN.  
It is named _Inception_ Distance because you’re using an Inception neural network (say, InceptionV3) for computing the distance. Here’s how you’ll do that, technically:
