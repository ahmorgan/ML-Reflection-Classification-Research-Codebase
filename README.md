# Codebase for my contributions to Fall 2024 OUR Research Scholars project + Spring 2025 Research Assistantship
# Fall Poster: Comparative Analysis of Resource-Efficient Multi-Label Text Classification Methods: GPT-4o vs fine-tuning with SetFit
# Spring paper: "A Comparison of Machine Learning Approaches for Student Challenge Classification" By Sandra Wiktor, Dr. Mohsen Dorodchi, Eunyoung Kim, Andrew Morgan, Nicole Wiktor, Frank Garcia
This repository contains all code created for the above projects. I could not include the train/test datasets I used for SetFit/GPT-4o or the raw data I used for train/test dataset construction for research ethics purposes.

Dependencies required (latest versions if not specified otherwise):
SetFit - setfit ver 1.0.3, huggingface-hub ver 0.23.5, transformers ver 4.43.3, torch/torchaudio/torchvision ver 2.4.1+cu124, optuna, numpy, sklearn |
GPT-4o - openai, numpy, sklearn |
Dataset Construction - numpy, pandas, openpyxl |
Data Visualization - matplotlib |
Disagreement Filter - nltk |
FastFit Implementation - fastfit, datasets ver 2.21.0, torch, numpy, sklearn, optuna, matplotlib

Fall Poster Abstract:

Text classification – the automatic pairing of text with labels – has proven itself a ubiquitous natural language processing task. The use of large language models (LLMs) fine-tuned on text data (in particular, BERT and its derivatives) is a common method for performing this task, as is prompting LLMs, such as OpenAI’s GPT models. One particular advancement in the problem of text classification has been the free SetFit (Sentence Fine-Tuning) framework pioneered by Hugging Face, shown to offset both training data requirements, only needing as little as eight examples per classification category to fit the data, and the computational cost that often comes with fine-tuning, while still providing competitive performance to more expensive methods. Notably, their results show that a fine-tuned ~100M parameter SetFit model surpasses GPT-3 by about 4.2% in accuracy. However, OpenAI has since released GPT-4o, which greatly improves upon GPT-3 and costs only about 14 cents per 100 classifications using our prompt. We seek to compare these two cost- and compute-efficient methods – GPT-4o and SetFit – of multi-label classification and examine SetFit two years after its release. We novelly compare the performance of a fine-tuned ~30M parameter SetFit model to GPT-4o on a multi-label classification task involving classifying student reflections from a software engineering course with one or more common course struggles – furthermore, we do so with the constraint that both methods of classification must be performable on average consumer-grade hardware (we use an 8GB NVIDIA GPU). We only fine-tune a ~30M parameter base model to meet this constraint. Our results show that, under our constraint, GPT-4o outperforms our SetFit model on macro-average F1 score across our label classes. We also find that our SetFit model is more likely to under-predict while GPT-4o is more likely to over-predict, noting limitations in our dataset construction methodology.

Fall Poster/UNCC CS Research Day Presentation:
![conceptualizingcobalt_48x36 (1)](https://github.com/user-attachments/assets/d3616a0f-46f8-4b0c-b29a-97a81b2ffbc9)

![1731715583413](https://github.com/user-attachments/assets/819f9954-bef7-4a78-82bc-7fe8876715f6)
