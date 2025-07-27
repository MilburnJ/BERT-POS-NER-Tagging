# Investigating Part-of-Speech Tagging and Named Entity Recognition with CRF and BERT Models

This project investigates and compares two popular NLP approaches—Conditional Random Fields (CRF) and BERT—for performing Part-of-Speech (POS) tagging and Named Entity Recognition (NER). The analysis evaluates the trade-offs in terms of accuracy, training time, and computational complexity using standard benchmark datasets.

## Objective

The primary objective is to analyze the performance differences between statistical models (CRF) and deep learning models (BERT) for two core NLP tasks:
- **POS Tagging** using the Penn Treebank corpus
- **NER** using the CoNLL-2003 English dataset

The notebook emphasizes understanding model behavior, accuracy, and resource usage across the two approaches.

## Tasks Performed

### 1. POS Tagging with CRF
- Implemented using `nltk.tag.CRFTagger`
- Trained on 3,000 sentences from the Penn Treebank corpus
- Evaluated using accuracy and training duration metrics

### 2. NER with CRF
- Used `sklearn-crfsuite` to build a CRF-based NER model
- Feature extraction included POS tags, word shapes, prefixes/suffixes, and capitalization
- Evaluated using F1 score and confusion matrix

### 3. NER with BERT
- Fine-tuned a pretrained BERT model from Hugging Face on CoNLL-2003
- Tokenized input using BERT tokenizer with attention masks
- Compared predictions with CRF model to analyze error patterns and performance gaps

## Results Summary

| Task                | Model | Metric         | Value     |
|---------------------|--------|----------------|-----------|
| POS Tagging         | CRF    | Accuracy       | ~95%      |
| NER                 | CRF    | F1 Score       | ~88%      |
| NER                 | BERT   | F1 Score       | >90%      |
| Training Time       | CRF    | Seconds        | Low       |
| Training Time       | BERT   | Minutes        | High      |

BERT demonstrated superior performance on the NER task, but with significantly higher computational cost. CRF remained competitive on POS tagging with fast training and strong accuracy.

## Requirements

- Python 3.7+
- Jupyter Notebook
- scikit-learn
- nltk
- sklearn-crfsuite
- transformers
- torch
- datasets
- seqeval
