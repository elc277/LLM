GPT Character-Level Language Model

This repository contains a transformer-based character-level language model built using PyTorch. The model is trained on Shakespeare's *Coriolanus* to learn and generate text in a similar literary style. The project includes:

- A custom GPT implementation (`train.py`)
- An evaluation toolkit for analyzing the generated text (`eval_functions.py`)
- A simpler Bigram baseline model (`BigramLM.py`)

---

## 📂 Project Structure
├── input.txt # Training corpus (Shakespeare's Coriolanus) 

├── train.py # Transformer-based GPT training script 

├── eval_functions.py # Perplexity, n-gram, sentiment, and entity analysis 

├── BigramLM.py # A simple bigram language model for comparison 

├── test.py # Code to check if the computer has cuda installed and working 

└── output.txt # Generated text after model inference 

## Model Overview

The main model in `train.py` is a GPT-style Transformer built from scratch with:

- Multi-head self-attention
- Positional and token embeddings
- 6 Transformer blocks
- Character-level tokenization
- Context window of 256 characters

### Training Details

| Parameter         | Value     |
|------------------|-----------|
| Batch Size       | 64        |
| Block Size       | 256       |
| Embedding Size   | 384       |
| Heads            | 6         |
| Layers           | 6         |
| Dropout          | 0.2       |
| Learning Rate    | 3e-4      |
| Iterations       | 5000      |
| Device           | GPU/CPU   |

The model is trained to predict the next character given a sequence of preceding characters.

---

## Evaluation & Analysis

`eval_functions.py` provides tools to evaluate the generated output on several fronts:

- **Perplexity Calculation**  
  Estimates model confidence in predictions.

- **Repeated N-gram Detection**  
  Detects redundancy in generation using top-10 repeated trigrams.

- **Sentiment Analysis**  
  Uses `TextBlob` for polarity scoring and `spaCy` for named entity recognition.

- **Visualization Plots**
  - `perplexity_comparison.png`
  - `ngram_repetition.png`
  - `sentiment_histogram.png`
  - `sentiment_trend.png`
  - `entity_frequency.png`

---

## Highlights

- **Model Checkpointing**  
  Automatically loads from or saves to `gpt_model_checkpoint.pth`.

- **Generation**  
  Post-training, the model generates ~5000 characters of text and writes it to `output.txt`.

- **Visualization**  
  After training, it plots the training and validation loss across intervals.

---

## Running the Code

### 1. Install Dependencies

```bash
pip install torch textblob spacy matplotlib
python -m textblob.download_corpora
python -m spacy download en_core_web_sm
```

### 2. Train the Model

```bash
python train.py
```
If a checkpoint exists, training resumes from it. Otherwise, it trains from scratch.

### 3. Evaluate the Output

```bash
python eval_functions.py
```
This will generate and save various plots and print analysis metrics.


### Future Improvements
Switch to subword or word-level tokenization

Add attention visualizations

Incorporate learning rate schedulers

Extend to multi-GPU or distributed training