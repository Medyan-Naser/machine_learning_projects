# Natural Language Classification Models

This repository demonstrates various machine learning models for text classification tasks, specifically sentiment analysis on restaurant reviews. We utilize different models such as Decision Trees, Random Forest, and Maximum Entropy (MaxEnt) to classify the reviews as either "positive" or "negative".

## Data

The dataset used is the **Restaurant_Reviews.tsv**, which contains reviews of restaurants and whether they were liked (positive or negative). The dataset is preprocessed by removing non-alphabetical characters, converting the text to lowercase, stemming words, and removing stopwords.

## Preprocessing

The preprocessing pipeline for all models includes the following steps:
1. **Text cleaning**: Remove non-alphabetical characters and convert the text to lowercase.
2. **Tokenization**: Split the review into individual words.
3. **Stemming**: Apply the Porter Stemmer to reduce words to their root form.
4. **Stopword Removal**: Remove common English stopwords like "the", "is", "and", etc.

## Models

### Decision Tree Classifier

The **Decision Tree Classifier** is a machine learning algorithm used for classification tasks. It builds a model based on a series of binary decisions, ultimately leading to a prediction.

### Random Forest Classifier

The **Random Forest Classifier** is an ensemble learning method that creates a forest of decision trees and combines their results to improve classification accuracy.


### Maximum Entropy (MaxEnt) Classifier

The **Maximum Entropy Classifier** uses a probabilistic approach to classify text. It aims to find the most "uniform" or least biased distribution over a set of possible classifications, given the constraints.

## Performance Evaluation

For each model, the following performance metrics are calculated:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The percentage of true positive predictions out of all positive predictions.
- **Recall**: The percentage of true positive predictions out of all actual positive instances.
- **F1 Score**: The harmonic mean of Precision and Recall, providing a balanced metric.

