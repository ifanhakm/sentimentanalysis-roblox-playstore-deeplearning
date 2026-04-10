# 🎮 Roblox User Review Sentiment Analysis using Deep Learning

This project implements Natural Language Processing (NLP) using Deep Learning approaches to classify user sentiments regarding the game **Roblox**. It was developed to achieve testing accuracy of 85%.

## 🎯 Project Objectives
* Build a robust Deep Learning model capable of classifying text reviews into three sentiment categories: **Positive, Negative, and Neutral**.
* Conduct experiments comparing different sequence modeling algorithms (*Bi-LSTM* and *GRU*) and data splitting proportions (*Train/Test Splits*).
* Mitigate *overfitting* by implementing *SpatialDropout1D*, *Early Stopping*, and *Learning Rate Reduction (ReduceLROnPlateau)*.

## 📊 Dataset & Preprocessing
The dataset consists of raw user text reviews from Google Play Store, where the user is located in Indonesia but using English to write their review. The data processing pipeline was intentionally streamlined to maximize Deep Learning performance and efficiency:
1. **Text Cleaning:** Removed URLs, numbers, punctuation, and non-alphabetic characters using *Regex*.
2. **Case Folding:** Converted all text to lowercase.
3. **Automated Labeling (VADER Lexicon):** Sentiment labels (Positive/Negative/Neutral) were extracted using the *compound score* from the VADER engine to establish an objective ground truth, bypassing the inherent bias and inconsistencies often found in human-generated star ratings.
    * `Compound >= 0.05` ➡️ Positive
    * `Compound <= -0.05` ➡️ Negative
    * `Other` ➡️ Neutral
4. **Tokenization & Padding:** Utilized the Keras `Tokenizer` with `max_vocab = 3000` and `maxlen = 50`.

## 🛠️ Experimental Scenarios & Architecture
This project evaluated three training scenarios to benchmark model stability and accuracy. The foundational architecture leverages an *Embedding Layer*, *SpatialDropout1D* (to prevent the model from overfitting to specific keywords), and a *Dense Layer* using a *Softmax* activation function.

| Scenario | Algorithm | Data Split (Train/Test) | Training Accuracy | Testing Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Scenario 1** | Bidirectional LSTM (Bi-LSTM) | 80% / 20% | ~94.6% | **85.98%** |
| **Scenario 2** | Gated Recurrent Unit (GRU) | 70% / 30% | ~96.2% | **85.85%** |
| **Scenario 3** | Bidirectional LSTM (Bi-LSTM) | 70% / 30% | ~95.0% | **84.42%** |

*Note: All models were trained utilizing `EarlyStopping` (monitoring `val_accuracy` with `mode='max'`) and `ReduceLROnPlateau` to ensure the preservation of optimal weights without succumbing to overfitting.*

## 💻 Usage (Inference)
The best-performing model (Scenario 1) can be deployed to predict novel, real-world sentences. Below is an example implementation:

```python
import tensorflow as tf
import numpy as np

# Load the Best Model
best_model = tf.keras.models.load_model('best_bilstm.h5')

# Example Inputs
sample_sentences = [
    "This game is an absolute masterpiece! I enjoy every second of it.",
    "Worst update ever, the lag is unbearable and it keeps crashing."
]

# Inference Loop (Assuming sentimen_prediksi, tok_s1, max_length, and le are predefined)
for text in sample_sentences:
    prediction = sentimen_prediksi(best_model, text, tok_s1, max_length, le)
    print(f"Text: '{text}' \nPrediction: {prediction}\n")
```

### Output
```
Text: 'This game is an absolute masterpiece! I enjoy every second of it.' 
Prediction: positive

Text: 'Worst update ever, the lag is unbearable and it keeps crashing.' 
Prediction: negative
```

## 📚 Libraries & Dependencies
Ensure your environment has the following dependencies installed:
- pandas & numpy
- matplotlib & seaborn
- scikit-learn
- tensorflow (Keras)
- nltk (Specifically for the VADER Lexicon)

## ✍️ Author
Ifan Hakim - AI Engineer
