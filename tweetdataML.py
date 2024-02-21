#libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the data
tweetdata = pd.read_csv (r"C:\Users\rashm\OneDrive\Desktop\streamlit\Datas\FinalBalancedDataset.csv")

#check for null values
tweetdata.isna().sum()

#check for class imbalance
toxic_count = len(tweetdata[tweetdata['Toxicity']==1])
normal_count = len(tweetdata[tweetdata['Toxicity']==0])
print(toxic_count,normal_count)

#Bag of Words (BoW)
#Splitting data into training and testing data
#seperate data and labels
textdata = tweetdata['tweet']
y = tweetdata['Toxicity']

# Create Bag of Words representation
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(textdata) 

#x holds data and y holds labels
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


#RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
#performing training
rfc.fit(x_train, y_train)
#testing
y_pred = rfc.predict(x_test)
#evaluation of model
print (metrics.classification_report(y_test, y_pred))
# confusion metris in detail
cm = metrics.confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data = cm, columns = ['Predicted Valid', 'Predicted Fraud'], index = ['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot = True, cmap = 'YlGnBu')
plt.show()
#ROC CURVE
#to get probability
probs= rfc.predict_proba(x_test)
preds=probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc= metrics.auc(fpr, tpr)
plt.title('ROC curve with Logistic Regression')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)
# Testing
y_pred = dt_classifier.predict(x_test)
# Evaluation of model
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Valid', 'Predicted Fraud'], index=['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = dt_classifier.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with Decision Tree Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(x_train, y_train)
# Testing
y_pred = nb_classifier.predict(x_test)
# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Non-Toxic', 'Predicted Toxic'], index=['Actual Non-Toxic', 'Actual Toxic'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = nb_classifier.predict_proba(x_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with Naive Bayes Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Specify the number of neighbors for KNN
n_neighbors = 5
# Initialize the K-NN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
# Train the K-NN Classifier
knn_classifier.fit(x_train, y_train)
# Testing
y_pred = knn_classifier.predict(x_test)
# Evaluation of model
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Valid', 'Predicted Fraud'], index=['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = knn_classifier.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with K-NN Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Initialize the SVM Classifier
svm_classifier = SVC()
# Train the SVM Classifier
svm_classifier.fit(x_train, y_train)
# Testing
y_pred = svm_classifier.predict(x_test)
# Evaluation of model
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Valid', 'Predicted Fraud'], index=['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = svm_classifier.decision_function(x_test)
fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with SVM Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

##################################################################################################

#TF-IDF representation
#Splitting data into training and testing data
#seperate data and labels
textdata = tweetdata['tweet']
y = tweetdata['Toxicity']

# Create TF-IDF representation
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(textdata)

#x holds data and y holds labels
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

#RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
#performing training
rfc.fit(x_train, y_train)
#testing
y_pred = rfc.predict(x_test)
#evaluation of model
print (metrics.classification_report(y_test, y_pred))
# confusion metris in detail
cm = metrics.confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data = cm, columns = ['Predicted Valid', 'Predicted Fraud'], index = ['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot = True, cmap = 'YlGnBu')
plt.show()
#ROC CURVE
#to get probability
probs= rfc.predict_proba(x_test)
preds=probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc= metrics.auc(fpr, tpr)
plt.title('ROC curve with Logistic Regression')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)
# Testing
y_pred = dt_classifier.predict(x_test)
# Evaluation of model
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Valid', 'Predicted Fraud'], index=['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = dt_classifier.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with Decision Tree Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(x_train, y_train)
# Testing
y_pred = nb_classifier.predict(x_test)
# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Non-Toxic', 'Predicted Toxic'], index=['Actual Non-Toxic', 'Actual Toxic'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = nb_classifier.predict_proba(x_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with Naive Bayes Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Train the K-NN Classifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
# Testing
y_pred = knn.predict(x_test)
# Evaluation of model
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Valid', 'Predicted Fraud'], index=['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = knn.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with K-NN Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

# Initialize the SVM Classifier
svm_classifier = SVC()
# Train the SVM Classifier
svm_classifier.fit(x_train, y_train)
# Testing
y_pred = svm_classifier.predict(x_test)
# Evaluation of model
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_mat = pd.DataFrame(data=cm, columns=['Predicted Valid', 'Predicted Fraud'], index=['Actual Valid', 'Actual Fraud'])
sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
plt.show()
# ROC Curve
probs = svm_classifier.decision_function(x_test)
fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.title('ROC curve with SVM Classifier')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()

