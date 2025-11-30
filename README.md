# Sentiment Analysis on Tamil–English Code-Mixed Social Media using NLP

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

o Dataset size: 14,841 social media posts with mixed Tamil and English text

o Sentiment classes: Positive, Negative, Mixed Feelings, Unknown State, Not-Tamil

o Emojis retained and processed as important affective features

# Architecture diagram 

![WhatsApp Image 2025-11-26 at 13 37 25_f24154e8](https://github.com/user-attachments/assets/913859de-e1f2-46fa-b08c-d1d78baa39b6)

# Proposed model

The proposed framework introduces a comprehensive sentiment analysis pipeline tailored for code-mixed Tamil-English (Tanglish) social media text enriched with emojis. The system is designed to address linguistic irregularities, transliteration variations and the affective cues conveyed through emojis, which significantly influence sentiment interpretation. The pipeline integrates advanced preprocessing techniques, multilingual transformer embeddings, class imbalance handling and multiple classification models to ensure robust performance across heterogeneous sentiment categories.

## A. Data Preprocessing

Each social media post undergoes systematic preprocessing to normalize noisy and informal user-generated content. The procedure includes lowercasing, removal of URLs, punctuation, special characters and numerals, along with elimination of stopwords. Emojis are preserved and processed as meaningful features due to their strong correlation with emotional expression in code-mixed text. This step ensures retention of affective signals while reducing textual noise.

## B. Label Encoding
To facilitate supervised learning, sentiment categories—Positive, Negative, Mixed Feelings, Unknown State, and Not-Tamil—are numerically encoded using scikit-learn’s LabelEncoder. This conversion allows machine learning and transformer-based models to interpret categorical outputs effectively.

## C. Feature Extraction

Two complementary feature extraction strategies are employed.
1.	TF-IDF Vectorization is applied to the preprocessed text for classical ML classifiers such as Logistic Regression and SVM.
2.	Contextual Transformer Embeddings are generated using multilingual models MuRIL and IndicBERT, which are capable of capturing semantic relationships, transliteration patterns and cross-lingual context present in Tamil-English code-mixed data.

## D. Class Imbalance Handling

The dataset contains 14,841 posts distributed across five sentiment categories, with a significant skew toward the Positive class. To mitigate class imbalance and alleviate model bias toward majority categories, Synthetic Minority Oversampling Technique (SMOTE) is applied at the feature representation stage. SMOTE generates synthetic samples for minority classes such as Mixed Feelings and Unknown State, enabling the classifiers to learn more generalized patterns. Emojis, retained as features, further enhance the representation of subtle emotional cues in minority classes.

## E. Data Splitting

The balanced feature set is partitioned into training and testing subsets using an 80:20 split. This ensures reliable performance evaluation while preventing overfitting and supporting generalization across unseen data.

## F. Model Training

Four distinct models are employed to evaluate performance across both classical and transformer-based paradigms:

•	Logistic Regression (TF-IDF features)

•	Support Vector Machine (SVM) (TF-IDF features)

•	MuRIL Transformer (multilingual contextual embeddings)

•	IndicBERT Transformer (Indian language–optimized embeddings)

Hyperparameters such as penalty terms, kernel functions and learning configurations are tuned to achieve optimal performance. The transformer models leverage deep multilingual embedding layers, providing enhanced capability to interpret transliterated text and mixed-language constructs.

## G. Prediction and Evaluation

Trained models generate sentiment predictions on the reserved test set. Performance is assessed using accuracy, precision, recall and F1-score, offering a comprehensive evaluation across imbalanced and nuanced sentiment classes. The integration of emoji sentiment cues, multilingual embeddings and synthetic oversampling collectively contributes to improved classification accuracy and robustness in code-mixed sentiment analysis.

# Results and Evaluation

The performance of the sentiment and emotion prediction system was evaluated using four models—Logistic Regression, SVM, MuRIL, and IndicBERT—each selected for their suitability in handling code-mixed Tamil-English (Tanglish) text. Classical machine learning models (Logistic Regression and SVM) rely on TF-IDF features, whereas transformer-based models (MuRIL and IndicBERT) leverage deep multilingual embeddings, enabling them to better interpret transliterated and emoji-rich content. The following figures present sample outputs generated by the system for real Tanglish user inputs.

### Example 1

**Input:** Nalla padam da sema feel varuthu 😊

**Sentiment:** Positive, **Emotion:** HAPPY

This example demonstrates the system’s ability to interpret both linguistic cues (“sema feel”) and emoji signals to classify the emotional tone accurately.

<img width="898" height="167" alt="image" src="https://github.com/user-attachments/assets/b1cbdc7b-c7e9-458f-aaee-76fc201067a7" />

### Example 2

**Input:** Intha trailer mokka da 😳

**Sentiment:** Negative,**Emotion:** SAD

Here, the system correctly identifies negative sentiment based on the lexical cue mokka (boring) and aligns the emotion with the user’s expression.
 
<img width="855" height="184" alt="image" src="https://github.com/user-attachments/assets/b97d18f7-7f35-48c9-944b-5d451eb9b2e5" />

### Example 3

**Input:** Super movie da love it 😍

**Sentiment:** Positive, **Emotion:** LOVE

This result highlights the model’s capability to detect emotion categories beyond basic polarity, distinguishing LOVE due to the combination of positive phrasing and emoji.

<img width="800" height="178" alt="image" src="https://github.com/user-attachments/assets/cae4f81e-1110-4185-9938-68b576aa08e2" />

### Example 4

**Input:** Ippadiye poguthu mass 🤣 ... Konjam Irmugan Madhiri irukku....All the best team..

**Sentiment:** Mixed_Feelings, **Emotion:** NEUTRAL 


This example illustrates the model’s effectiveness in handling complex or multi-topic expressions, where mixed sentiment is present within a single message.

<img width="970" height="150" alt="image" src="https://github.com/user-attachments/assets/4ce12e76-2dea-4438-b9d4-1d9be3f81dda" />

### Evaluation

The evaluation phase assesses the effectiveness of the four selected models—Logistic Regression, SVM, MuRIL, and IndicBERT—using standard performance metrics, including accuracy, precision, recall, and F1-score. These metrics provide a comprehensive understanding of each model’s capability in handling code-mixed Tamil-English sentiment classification.

<img width="784" height="227" alt="image" src="https://github.com/user-attachments/assets/ad4ad569-3e16-44ee-88e3-2470f8ba9d07" />

#### Comparative Analysis of Models

<img width="752" height="452" alt="image" src="https://github.com/user-attachments/assets/860a8ed6-5501-42ac-93ab-6a447e3d6aff" />

The above graph summarizes the quantitative results obtained from all models. Logistic Regression demonstrated the highest performance across all metrics, achieving an accuracy of 90.23% and an F1-score of 89.67%. SVM also performed strongly, with an accuracy of 87.8% and an F1-score of 87.6%, reflecting consistent generalization across sentiment categories.

IndicBERT, a multilingual transformer model, achieved 78.88% accuracy and a 75% F1-score, outperforming MuRIL but still trailing behind classical machine learning models. MuRIL recorded lower metrics (accuracy 67%, F1-score 60%), indicating challenges in handling highly transliterated and code-mixed inputs without further domain-specific adaptation.

Overall, the results show that classical models—when paired with TF-IDF features—still deliver superior performance in structured sentiment tasks involving code-mixed text. Transformer-based models demonstrate potential but require deeper fine-tuning and expanded domain-relevant corpora to reach competitive performance on Tanglish datasets.
Accuracy Comparison
 
#### Accuracy Comparison Across Models

<img width="660" height="376" alt="image" src="https://github.com/user-attachments/assets/482390d3-3557-4d31-bb63-71b4a37fe844" />

The above graph illustrates the accuracy distribution across the four models. Logistic Regression emerged as the top performer with 90.23% accuracy, while SVM closely followed with 87.8%. IndicBERT achieved 78.88%, showing reasonable proficiency in handling multilingual and transliterated text. MuRIL recorded the lowest accuracy (67%), highlighting limitations in zero-shot or minimally fine-tuned transformer settings for code-mixed content.
These results reinforce that traditional algorithms maintain competitive performance when supported by strong feature engineering, whereas transformers require additional adaptation to handle informal and irregular text patterns.
F1-Score Comparison
 
#### F1-Score Comparison Across Model

<img width="480" height="288" alt="image" src="https://github.com/user-attachments/assets/5e5e526f-d261-4e06-ac82-195ecf8d72eb" />

The above graph compares the F1-scores across the models. Logistic Regression again leads with an F1-score of 89.67%, reflecting high reliability in distinguishing sentiment categories—even with imbalanced data. SVM achieved 87.6%, indicating robust sentence-level generalization.
IndicBERT scored 75%, performing significantly better than MuRIL (60%), but still falling short of classical ML models. These results emphasize that while transformer models exhibit strong multilingual capabilities, their performance on code-mixed Tanglish text remains sensitive to domain size, quality of embeddings, and the transliteration patterns present in the training data.

Through quantitative evaluation, the following insights were observed:

•	Classical ML models outperform transformers for the given Tanglish dataset due to the structured nature of TF-IDF features and the relatively small dataset size.

•	IndicBERT shows potential, benefiting from multilingual pretraining across Indian languages, making it more suitable than MuRIL for code-mixed Tamil-English input.

•	MuRIL underperforms likely because of insufficient exposure to transliterated Tamil forms and high variability in user-generated text.

•	Transformer models require domain-specific augmentation, larger training corpora, and optimized fine-tuning to fully leverage their contextual encoding abilities.

In summary, Logistic Regression and SVM deliver the most reliable results for Tanglish sentiment analysis under current conditions, while IndicBERT presents a promising foundation for future multilingual transformer improvements.
