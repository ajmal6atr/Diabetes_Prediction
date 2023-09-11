'''Context
This dataset is originally from the National Institute of Diabetes and Digestive 
and Kidney Diseases. The objective of the dataset is to diagnostically predict 
whether or not a patient has diabetes, based on certain diagnostic measurements 
included in the dataset. Several constraints were placed on the selection of these
 instances from a larger database. In particular, all patients here are females
 at least 21 years old of Pima Indian heritage.

Content
The datasets consists of several medical predictor variables and one 
target variable, Outcome. Predictor variables includes the number of pregnancies 
the patient has had, their BMI, insulin level, age, and so on.
https://www.kaggle.com/uciml/pima-indians-diabetes-database'''

import pandas as pd

# load dataset

diabete = pd.read_csv("diabetes.csv")

diabete.head()

diabete.index

diabete.columns

#split dataset in features and target variable
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']

X = diabete[feature_cols] # Features
y = diabete.Outcome # Target variable
X
y
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()
X_train
y_train
# fit the model with data
logreg.fit(X_train,y_train)

#Prediction
y_pred=logreg.predict(X_test)
y_pred
y_test

# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("Confusion matrix",cnf_matrix)
print("misclassified:%d" ,(y_test!=y_pred).sum())
print("classified:%d" ,(y_test==y_pred).sum())
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score",metrics.f1_score(y_test, y_pred))
#output
new1= logreg.predict([[6,148,72,35,0,33.6,0.627,50]])
print(new1)

if new1==1:
    print('Diabetes Positive')
else:
    print('Diabetes negative')
    
    
    
new=logreg.predict([[1,85,66,29,0,26.6,0.35100000000000003,31]])
print(new)
if new==1:
    print('Diabetes Positive')
else:
    print('Diabetes negative')



from sklearn.neighbors import KNeighborsClassifier

KNN_classifier=KNeighborsClassifier(n_neighbors=7)

KNN_classifier.fit(X_train,y_train)

y_pred=KNN_classifier.predict(X_test)


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("KNN:Confusion matric",cnf_matrix)
print("KNN:misclassified:%d" ,(y_test!=y_pred).sum())
print("KNN:classified:%d" ,(y_test==y_pred).sum())
print("KNN:Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("KNN:Precision:",metrics.precision_score(y_test, y_pred))
print("KNN:Recall:",metrics.recall_score(y_test, y_pred))
print("KNN:F1 score",metrics.f1_score(y_test, y_pred))