# tweedataML
This code performs text classification on a dataset of tweets using various machine learning models like Random Forest, Decision Trees, Naive Bayes, K-NN, and SVM. It employs Bag of Words (BoW) and TF-IDF representations for feature extraction.
Overview
This repository contains code for classifying the toxicity of tweets using machine learning models. It utilizes techniques like Bag of Words (BoW) and TF-IDF for feature extraction and trains classifiers such as Random Forest, Decision Trees, Naive Bayes, K-NN, and SVM for prediction.

Dataset
The dataset consists of tweets labeled as toxic or non-toxic. It is preprocessed to handle null values and check class imbalance.

Dependencies
Python 3
Libraries: pandas, scikit-learn, matplotlib, seaborn
Usage
Clone the repository:
git clone https://github.com/username/repository.git
Install dependencies:
pip install -r requirements.txt
Load the dataset and preprocess it to handle null values and check for class imbalance.

Choose between Bag of Words (BoW) or TF-IDF representation for feature extraction.

Train and test different machine learning models provided in the code.

Evaluate model performance using classification reports, confusion matrices, and ROC curves.

Results
The code provides detailed performance metrics for each machine learning model tested, facilitating comparison and selection of the best-performing model.
