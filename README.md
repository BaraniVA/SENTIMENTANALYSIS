# Sentiment Analysis Using Random Forest and Naive Bayes

## Overview

This project focuses on performing sentiment analysis on app reviews using two machine learning models: **Random Forest** and **Naive Bayes**. The goal is to classify user reviews into three sentiment categories: Positive, Neutral, and Negative. The project also includes a hybrid model that combines the predictions of both classifiers to improve accuracy.

## Project Structure

 **Data Loading and Exploration**: Load the dataset and perform an initial exploration to understand its structure.
 **Data Preprocessing**: Clean the data, handle missing values, and label sentiments based on the review scores.
 **Exploratory Data Analysis (EDA)**: Visualize the data to gain insights into sentiment distribution, review length, and word frequency.
 **Feature Extraction**: Use the TF-IDF vectorizer to transform the text data into numerical features suitable for model training.
 **Model Training**: Train the **Random Forest** and **Naive Bayes** classifiers on the dataset.
 **Model Evaluation**: Evaluate the performance of both models using metrics such as accuracy, classification reports, and confusion matrices.
 **Hybrid Model**: Develop a hybrid algorithm that combines the predictions of both classifiers, incorporating confidence levels to refine the predictions.
 **Model Comparison**: Compare the accuracy and performance of the Random Forest, Naive Bayes, and Hybrid models.
 **Findings**: Analyze common sentiments, app popularity, and competitor comparisons.

## Installation

To run this project, you'll need to install the following Python libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn wordcloud joblib
```

## Usage

1. **Load the Dataset**: Ensure your dataset (`app_reviews.csv`) is placed in the specified directory.
2. **Run the Code**: Execute the script to load, preprocess, and analyze the data.
3. **Train Models**: The code will automatically train the Random Forest, Naive Bayes, and Hybrid models.
4. **Evaluate Models**: View the evaluation metrics and confusion matrices to assess model performance.
5. **Test Hybrid Algorithm**: Input your own review to see how the hybrid model classifies its sentiment.

## Key Functions and Methods

 `label_sentiment(score)`: Labels the sentiment based on the review score.
 `hybrid_algorithm(input_text)`: Combines Random Forest and Naive Bayes predictions to determine the sentiment with confidence levels.
 `calculate_hybrid_accuracy(X_test, y_test)`: Computes the accuracy of the hybrid model on the test set.
 `test_user_input_sentiment(input_text)`: Allows you to test the sentiment of custom review input using the hybrid model.

## Results

 **Random Forest Accuracy**: ~64%
 **Naive Bayes Accuracy**: ~58%
 **Hybrid Model Accuracy**: ~88%

The hybrid model outperformed individual classifiers by leveraging their strengths, resulting in higher accuracy and better sentiment prediction.

## Visualization

 **Sentiment Distribution**: Understand the distribution of sentiments across the dataset.
 **Word Cloud**: Visualize the most common words in reviews.
 **Confusion Matrices**: Evaluate model predictions against actual sentiments.
 **Model Accuracy Comparison**: Compare the accuracy of different models using bar charts.

## Findings

 Detailed insights into the most common sentiments, popular apps, and a comparison of competitor apps based on review sentiments and ratings.

## Conclusion

This project demonstrates the effectiveness of combining multiple machine learning models to improve sentiment analysis. The hybrid model achieved a significant accuracy boost, making it a robust solution for analyzing user sentiments in app reviews.

