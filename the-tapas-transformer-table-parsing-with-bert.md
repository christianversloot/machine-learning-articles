---
title: "The TAPAS Transformer: Table Parsing with BERT"
date: "2021-03-05"
categories: 
  - "buffer"
  - "deep-learning"
tags: 
  - "deep-learning"
  - "machine-learning"
  - "nlp"
  - "table-parsing"
  - "tapas"
  - "transformer"
  - "transformers"
---

TAPAS (Table Parser) is a weakly supervised Transformer-based question answering model that reasons over tables _without_ generating logical forms. Instead, it predicts a minimal program by selecting a relevant subset of table cells + the most likely aggregation operator to be executed on top of these cells, jointly. This allows TAPAS to learn operations based on natural language without requiring some explicit formalism.
