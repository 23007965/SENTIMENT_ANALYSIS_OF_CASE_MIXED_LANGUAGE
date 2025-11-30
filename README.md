# SENTIMENT_ANALYSIS_OF_CASE_MIXED_LANGUAGE

# Project Overview

This project addresses the challenging task of sentiment and emotion analysis in Tamil–English code-mixed social media posts enriched with emojis. Code-mixing, transliteration variations, informal grammar, and emoji usage complicate traditional sentiment analysis approaches. Our system integrates advanced preprocessing, TF-IDF features, transformer-based embeddings (MuRIL, IndicBERT), and class-imbalance handling via SMOTE to accurately classify sentiments and emotions in this multilingual, informal context.

# Features

1. Supports Tamil–English (Tanglish) code-mixed text and emoji-rich posts

2. Robust text preprocessing including URL removal, normalization, stopword elimination, and emoji preservation

3. Classical ML models using TF-IDF: Logistic Regression, SVM

4. Transformer-based models: MuRIL and IndicBERT for contextual embeddings

5. Synthetic Minority Oversampling Technique (SMOTE) for class imbalance handling

6. Comprehensive evaluation metrics: accuracy, precision, recall, and F1-score

# Dataset Description

. Dataset size: 14,841 social media posts with mixed Tamil and English text

. Sentiment classes: Positive, Negative, Mixed Feelings, Unknown State, Not-Tamil

. Emojis retained and processed as important affective features

