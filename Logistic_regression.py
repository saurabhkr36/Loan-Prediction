# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:34:52 2020

@author: Heisenberg
"""

import pandas as pd
import numpy as np

test=pd.read_csv('test_lAUu6dG.csv')

train=pd.read_csv('train_ctrUa4K.csv')
loan_id=test['Loan_ID']
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



from sklearn.linear_model import LogisticRegression
svc=LogisticRegression()
svc.fit(train.iloc[:,:-1].values,train['Loan_Status'])
y_pred=svc.predict(test)

'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=256, criterion='gini', random_state = 0)
classifier.fit(train.iloc[:,:-1].values,  train['Loan_Status'])
y_pred=classifier.predict(test)
'''
# Predicting the Test set results
predictions=[]
for i in range(len(y_pred)):
    if(y_pred[i]==1):
        predictions.append("Y")
    else:
        predictions.append("N")
predictions=pd.DataFrame(predictions)
output=pd.concat([loan_id,predictions],axis=1)
output=output.rename(columns={0:'Loan_Status'})
output.set_index('Loan_ID',inplace=True)
soln=output.to_csv("Final_Final.csv")