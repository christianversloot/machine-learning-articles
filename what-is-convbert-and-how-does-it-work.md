---
title: "What is ConvBERT and how does it work?"
date: "2021-02-26"
categories: 
  - "buffer"
  - "deep-learning"
tags: 
  - "bert"
  - "convbert"
  - "huggingface"
  - "nlp"
  - "transformer"
  - "transformers"
---

Convolutional BERT (ConvBERT) improves the original BERT by replacing some Multi-headed Self-attention segments with cheaper and naturally local operations, so-called span-based dynamic convolutions. These are integrated into the self-attention mechanism to form a mixed attention mechanism, allowing Multi-headed Self-attention to capture global patterns; the Convolutions focus more on the local patterns, which are otherwise captured anyway. In other words, they reduce the computational intensity of training BERT.
