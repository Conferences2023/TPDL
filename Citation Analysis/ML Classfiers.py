
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load data from xlsx file
data = pd.read_excel('/content/sample_data/Annotation-MU.xlsx')
# Split data into features (X) and labels (y)
X = data["CitationContext"]
y = data["Class"]

# Convert text into bi-gram vectors
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_clf = SVC(kernel="linear")
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", svm_acc)
print(classification_report(y_test, svm_pred))

# Train Logistic Regression classifier
logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
logreg_pred = logreg_clf.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_pred)
print("Logistic Regression Accuracy:", logreg_acc)
print(classification_report(y_test, logreg_pred))

# Train Naive Bayes classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
nb_pred = nb_clf.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_acc)
print(classification_report(y_test, nb_pred))
