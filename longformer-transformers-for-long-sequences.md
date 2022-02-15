---
title: "Longformer: Transformers for Long Sequences"
date: "2021-03-11"
categories: 
  - "buffer"
  - "deep-learning"
tags: 
  - "deep-learning"
  - "language-model"
  - "longformer"
  - "machine-learning"
  - "nlp"
  - "transformer"
---

Transformers have really changed the NLP world, in part due to their self-attention component. But this component is problematic in the sense that it has quadratic computational and memory growth with sequence length, due to the QK^T diagonals (Questions, Keys diagonals) in the self-attention component. By consequence, Transformers cannot be trained on really long sequences because resource requirements are just too high. BERT, for example, sets a maximum sequence length of 512 characters.
