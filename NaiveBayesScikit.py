from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



#Load Dataset
data = load_breast_cancer()

#Organise our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

gnb = GaussianNB()
model = gnb.fit(train, train_labels)

#Make Predictions

preds = gnb.predict(test)

print('Accuracy with Naive Bayes is', accuracy_score(test_labels, preds))


rf = RandomForestClassifier()
model = rf.fit(train, train_labels)

#Make Predictions

preds = rf.predict(test)

print('Accuracy with Random Forest Classifier is', accuracy_score(test_labels, preds))

dt = DecisionTreeClassifier()
model = dt.fit(train, train_labels)

#Make Predictions

preds = dt.predict(test)

print('Accuracy with Decision tree classifier is', accuracy_score(test_labels, preds))