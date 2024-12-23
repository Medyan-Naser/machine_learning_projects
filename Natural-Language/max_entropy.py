# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import ssl
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import MaxentClassifier

# SSL fix for nltk downloader
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')

# Importing the dataset
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'Restaurant_Reviews.tsv')
dataset = pd.read_csv(file_path, delimiter = '\t', quoting = 3)

# Preprocessing the data
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    if dataset['Liked'][i] == 0:
        result = "negative"
    else:
        result = "positive"
    
    # Adding features to the corpus (each word is treated as a feature)
    for word in review:
        corpus.append(({'item': word}, result))

# Splitting the dataset into Training and Test set
X_train, X_test = train_test_split(corpus, test_size=0.20, random_state=0)

# Training the Maximum Entropy Classifier
classifier = MaxentClassifier.train(X_train, algorithm='GIS', max_iter=100)

# Predicting the results on the test set
y_true = [label for (_, label) in X_test]
y_pred = [classifier.classify(features) for (features, _) in X_test]

# Making the Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Accuracy, Precision, Recall, F1 Score
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
f1_score = 2 * precision * recall / (precision + recall)

# Print Metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Plotting the Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Adding labels
classes = ['Positive', 'Negative']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Adding text annotations
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Plotting Accuracy, Precision, Recall, and F1 Score
metrics = [accuracy, precision, recall, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(8, 5))
plt.bar(metric_names, metrics, color=['skyblue', 'orange', 'lightgreen', 'lightcoral'])
plt.title('Classification Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()
