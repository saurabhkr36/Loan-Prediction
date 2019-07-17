# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:37:02 2019

@author: Heisenberg
"""

import pandas as pd
import numpy as np

test=pd.read_csv('test_Y3wMUE5_7gLdaTN.txt')

train=pd.read_csv('train_u6lujuX_CVtuZ9i.txt')

#describes columns with int values stats
describe=test.describe()
del train['Loan_ID']
del test['Loan_ID']


#helps eliminating nan values in a specific column
test['Loan_Amount_Term'].fillna('360',inplace = True)
test['Dependents'].fillna( 'a' ,inplace = True)

test['Gender'].fillna('M',inplace = True)
test['Self_Employed'].fillna('No',inplace = True)
test['LoanAmount'].fillna(np.mean(test['LoanAmount']),inplace=True)
test['Credit_History'].fillna(np.random.random_integers(0,1),inplace=True)


train.dropna(subset=['Gender','Married','Dependents','Loan_Amount_Term'],inplace = True)
train['Self_Employed'].fillna('No',inplace = True)
train['LoanAmount'].fillna(np.mean(train['LoanAmount']),inplace=True)
train['Credit_History'].fillna(np.random.random_integers(0,1),inplace=True)



test.apply(lambda x: sum(x.isnull()))
train.apply(lambda x: sum(x.isnull()))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
train['Gender'] = labelencoder_x.fit_transform(train['Gender'])
train['Married'] = labelencoder_x.fit_transform(train['Married'])
train['Dependents'] = labelencoder_x.fit_transform(train['Dependents'])
train['Education'] = labelencoder_x.fit_transform(train['Education'])
train['Self_Employed'] = labelencoder_x.fit_transform(train['Self_Employed'])
train['Property_Area'] = labelencoder_x.fit_transform(train['Property_Area'])
train['Loan_Status'] = labelencoder_x.fit_transform(train['Loan_Status'])

test['Gender'] = labelencoder_x.fit_transform(test['Gender'])
test['Married'] = labelencoder_x.fit_transform(test['Married'])
test['Dependents'] = labelencoder_x.fit_transform(test['Dependents'])
test['Education'] = labelencoder_x.fit_transform(test['Education'])
test['Self_Employed'] = labelencoder_x.fit_transform(test['Self_Employed'])
test['Property_Area'] = labelencoder_x.fit_transform(test['Property_Area'])


#Model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(train.iloc[:,:-1].values,  train['Loan_Status'])


# Predicting the Test set results
y_pred = classifier.predict(test)

y_pred=pd.DataFrame(y_pred)


submission =y_pred
submission.to_csv("final_submission.csv")

Loan_ID = test['Loan_ID']
Loan_ID.to_csv("Loan_ID.csv")
