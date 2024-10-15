import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings

warnings.filterwarnings('ignore')


data = pd.read_csv('Employee_Attrition.csv')
print(data.head())
print(data.isnull().any())

def sal(x):
    if x < 7000:
        return 'Low'
    elif x >= 7000 and x < 14000:
        return 'Medium'
    else:
        return 'High'

data['MonthlyIncome'] = data['MonthlyIncome'].apply(sal)

le = LabelEncoder()
data['JobRole'] = le.fit_transform(data['JobRole'])
data['OverTime'] = le.fit_transform(data['OverTime'])
data['BusinessTravel'] = le.fit_transform(data['BusinessTravel'])
data['Attrition'] = le.fit_transform(data['Attrition'])
data['MonthlyIncome'] = le.fit_transform(data['MonthlyIncome'])  # High=0,Low=1,Medium=2
data['Gender'] = le.fit_transform(data['Gender'])



ip = ['Age', 'Gender','JobRole','JobLevel','JobSatisfaction', 'MonthlyIncome',
      'OverTime','BusinessTravel', 'NumCompaniesWorked','PercentSalaryHike', 'YearsAtCompany','TotalWorkingYears']
op = ['Attrition']

scalar = StandardScaler()
data[ip] = scalar.fit_transform(data[ip])
data.head()

X = data[ip]
Y = data[op]

turnover_rate = data['Attrition'].value_counts()
print(turnover_rate)
print(data.groupby('Attrition').mean())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)
y_pred_DT = DT.predict(X_test)
acc_DT = accuracy_score(y_pred_DT,Y_test)*100
print("Accuracy of Decision Tree is:",acc_DT)

NB = GaussianNB()
NB.fit(X_train, Y_train)
y_pred_NB = NB.predict(X_test)
acc_NB = accuracy_score(y_pred_NB,Y_test)*100
print("Accuracy of Naive Bayes is:",acc_NB)

RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
y_pred_RF = RF.predict(X_test)
acc_RF = accuracy_score(y_pred_RF,Y_test)*100
print("Accuracy of Random Forest is:",acc_RF)

LR = LogisticRegression()
LR.fit(X_train, Y_train)
y_pred_LR = LR.predict(X_test)
acc_LR = accuracy_score(y_pred_LR,Y_test)*100
print("Accuracy of Logistic Regression is:",acc_LR)

final = LR.predict([[41,0,7,2,4,0,1,2,8,11,6,8]])
print(final)

final = LR.predict([[49,1,6,2,2,0,0,1,1,23,10,10]])
print(final)
"""
41,0,7,2,4,0,1,2,8,11,6,8 -> yes(1)
49,1,6,2,2,0,0,1,1,23,10,10 -> no(0)
"""
pickle.dump(RF, open('model.pkl','wb')) 

#write binary, rb= read binary