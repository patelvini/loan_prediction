# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

train  = pd.read_csv('train_loan.csv')
test = pd.read_csv('test_loan.csv')

train_original = train.copy() 
test_original = test.copy()

'''
print(train.columns)
print(test.columns)


print(train.shape)
print(test.shape)

print(train['Loan_Status'].value_counts())


# Normalize can be set to True to print proportions instead of number
print(train['Loan_Status'].value_counts(normalize = True))

train['Loan_Status'].value_counts().plot.bar()
plt.show()


# Independent Variable (Categorical)

plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()
 
# Independent Variable (Ordinal)

plt.subplot(131) 
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 
plt.subplot(132) 
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(133) 
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
plt.show()

# Independent Variable (Numerical)

plt.subplot(121) 
sns.distplot(train['ApplicantIncome']); 
plt.subplot(122) 
train['ApplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()


Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
'''


# Missing Value 
# Let’s list out feature-wise count of missing values.

# print(train.isnull().sum())


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


#print(train['Loan_Amount_Term'].value_counts())

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#print(train.isnull().sum())


# Outlier Treatment
# One way to remove the skewness is by doing the log transformation. 

train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20) 
test['LoanAmount_log'] = np.log(test['LoanAmount'])



# Evaluation Metrics for Classification Problems

# Model Building : Part I



#Lets drop the Loan_ID variable as it do not have any effect on the loan status.
#We will do the same changes to the test dataset which we did for the training dataset.

train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)

#Sklearn requires the target variable in a separate dataset. 
#So, we will drop our target variable from the train dataset and save it in another dataset.

X = train.drop('Loan_Status',1) 
y = train.Loan_Status











































