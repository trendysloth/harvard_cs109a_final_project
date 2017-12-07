---
title: Baseline Ridge Model
notebook: baseline-ridge-model.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
import seaborn as sns
%matplotlib inline
```




```python
df_user = pd.read_csv("Montréal_users.csv")
df_review= pd.read_csv("Montréal_reviews.csv")
df_business = pd.read_csv("Montréal_businesses.csv")

msk = np.random.rand(len(df_review)) < 0.75
df_review_train = df_review[msk]
df_review_test = df_review[~msk]

df_review_business_id_train = list(df_review_train['business_id'])
df_review_user_id_train = list(df_review_train['user_id'])
df_business_train = df_business[df_business['business_id'].isin(df_review_business_id_train)]
df_user_train = df_user[df_user['user_id'].isin(df_review_user_id_train)]

df_review_business_id_test = list(df_review_test['business_id'])
df_review_user_id_test = list(df_review_test['user_id'])
df_business_test = df_business[df_business['business_id'].isin(df_review_business_id_test)]
df_user_test = df_user[df_user['user_id'].isin(df_review_user_id_test)]
```




```python
print(df_review_train.shape)
print(len(df_review_train['user_id'].unique()))
print(len(df_review_train['business_id'].unique()))

print(df_user_train.shape)
print(len(df_user_train['user_id'].unique()))

print(df_business_train.shape)
print(len(df_business_train['business_id'].unique()))
```


    (30999, 10)
    3201
    2429
    (3201, 24)
    3201
    (2429, 17)
    2429




```python
df_review_train.to_csv("Montreal_review_train.csv")
df_review_test.to_csv("Montreal_review_test.csv")
df_user_train.to_csv("Montreal_user_train.csv")
df_user_test.to_csv("Montreal_user_test.csv")
df_business_train.to_csv("Montreal_business_train.csv")
df_business_test.to_csv("Montreal_business_test.csv")
```




```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
```




```python
def RMSE(real, predicted):
    """
    Calculate the root mean squared error between a matrix of real ratings and predicted ratings
    :param real: A matrix containing the real ratings (with 'NaN' for any missing elements)
    :param predicted: A matrix of predictions
    :return: The RMSE as a float
    """
    return np.sqrt(np.nanmean(np.square(real - predicted)))
```




```python
def predict(df, df_total):
    #get unique user and business id's
    unique_ids = list(set(df_total['user_id'].as_matrix()))
    unique_businesses = list(set(df_total['business_id'].as_matrix()))
    mapping_users = {}
    mapping_businesses = {}
    ct=0
    for item in unique_ids:
        mapping_users[item] = ct
        ct+=1
    ct=0
    for item in unique_businesses:
        mapping_businesses[item] = ct
        ct+=1

    A = np.zeros([df.shape[0],1+len(unique_ids)+len(unique_businesses)])

    A[:,0] = 1
    groundtruth = []
    ct = 0
    for index, row in df.iterrows():
        user_id = row['user_id']
        business_id = row['business_id']
        user_index = mapping_users[user_id]
        business_index = mapping_businesses[business_id]
        A[ct,1+user_index]=1
        A[ct,1+len(unique_ids)+business_index]=1
        groundtruth.append(row['stars'])
        ct+=1
    return (A, groundtruth)

x_train = predict(df_review_train, df_review)[0]
y_train = predict(df_review_train, df_review)[1]
x_test = predict(df_review_test, df_review)[0]
y_test = predict(df_review_test, df_review)[1]

model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100])
model.fit(x_train, y_train)
model.alpha_

model = Ridge(fit_intercept=False, alpha = model.alpha_)

model = model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

print("Train RMSE score is {}".format(RMSE(y_train, y_pred_train)))
print("Test RMSE score is {}".format(RMSE(y_test, y_pred_test)))
```

