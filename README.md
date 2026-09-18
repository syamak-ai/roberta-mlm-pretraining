# roberta-mlm-from-scratch

A small RoBERTa-based masked language model trained from scratch on the writings of Immanuel Kant.

The project explores the process of building a domain-specific language model, from training a custom Byte-Level BPE tokenizer to pretraining a RoBERTa architecture with a masked language modeling objective.

## Overview

The goal of this project is to build a compact Transformer language model specialized in Kant's writing style and vocabulary.
The pipeline consists of:

- Training a custom Byte-Level BPE tokenizer

- Configuring the tokenizer for RoBERTa

- Building a small RoBERTa Transformer from scratch

- Preparing Kant's writings as a PyTorch dataset

- Pretraining the model using Masked Language Modeling (MLM)

- Evaluating the model using validation loss 

- Using the trained model for masked-token prediction

## Model Architecture
The model is based on the RoBERTa architecture and was configured as a relatively small model to make training feasible on limited hardware.
A custom Byte-Level BPE tokenizer is trained directly on the Kant corpus using the Hugging Face tokenizers library.

## Training

The model is trained using Masked Language Modeling.
During training, approximately 15% of tokens are selected for masking. The model learns to predict the original tokens from their surrounding context.

