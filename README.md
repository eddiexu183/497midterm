# Headline Generation Using an LSTM-based Sequence-to-Sequence Model

## Project Overview
This project implements a **sequence-to-sequence (seq2seq) model with LSTM** to generate **news headlines** from longer news articles. The model is trained using **Keras** and **TensorFlow** on a dataset of summarized news articles.

## Dataset
- The dataset used is **news_summary.csv**.
- It contains:
    - `ctext`: The cleaned article text.
    - `headlines`: The corresponding headline.
- Preprocessing includes **lowercasing, filtering long sequences, and tokenization**.

## Model Architecture
- **Encoder:** An LSTM processes the input text and generates a context vector.
- **Decoder:** Another LSTM generates the headline token by token, initialized with the encoder’s final states.
- **Embedding Layers:** Learn vector representations for words.
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam`