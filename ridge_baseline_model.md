---
nav_include: 2
title: Ridge Model Baseline
notebook: ridge_baseline_model.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}



```python
#Set up workspace
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
import seaborn as sns
%matplotlib inline
import os
import json
from datetime import datetime
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from statsmodels.tools import add_constant

```




```python
#Read in files
def read_city(city):
    businesses = pd.read_csv(str(city)+'_businesses.csv')
    users = pd.read_csv(str(city)+'_users.csv')
    reviews_orig = pd.read_csv(str(city)+'_reviews.csv')
    
    reviews_df = reviews_orig.pivot(index = 'user_id', columns ='business_id', values = 'stars') # used to have : .fillna(0)
    
    return {'businesses': businesses, 'users': users, 'reviews_df': reviews_df, 'original reviews': reviews_orig}


Montreal_dfs = read_city('Montréal')

businesses = Montreal_dfs['businesses']

users = Montreal_dfs['users']
users['yelping_since'] = [datetime.strptime(i, '%Y-%m-%d') for i in users['yelping_since']]
users['indexed_id'] = range(1, len(users) + 1)

orig_reviews = Montreal_dfs['original reviews']

reviews_df = Montreal_dfs['reviews_df']
```




```python
#Cleaning and merging

#drop unnecessary columns
businesses = businesses.drop('Unnamed: 0', 1)
users = users.drop('Unnamed: 0', 1)
orig_reviews = orig_reviews.drop('Unnamed: 0', 1)

#Rename columns to prevent duplicates in merged dataframe
businesses = businesses.rename(columns={'stars': 'business_stars','name':'business_name','review_count':'business_review_count'})
orig_reviews = orig_reviews.rename(columns={'cool':'review_cool','date':'review_date','funny':'review_funny','useful':'review_useful'})
users = users.rename(columns={'cool':'user_cool_count','fan':'user_fans','friends':'user_friends','funny':'user_funny_count','name':'user_name','review_count':'user_review_count','useful':'user_useful_count'})

#Merging datasets
df_1 = pd.merge(orig_reviews, users, on='user_id')
df_total = pd.merge(df_1, businesses, on='business_id')
df_total = df_total.drop('business_stars',1) #Drop columns of values that must be calculated endogenously within train and test sets
df_total = df_total.drop('average_stars',1)
```




```python
#Formulas to return baseline scores of individual businesses and users

def business_baseline(train_df,business_id,business_total_avg_stars):
    average_stars = np.average(train_df['stars'], weights=(train_df['business_id']==business_id))
    divergence = average_stars - business_total_avg_stars

    return divergence

def user_baseline(train_df,user_id,user_total_avg_stars):
    average_stars = np.average(train_df['stars'], weights=(train_df['user_id']==user_id))
    divergence = average_stars - user_total_avg_stars   

    return divergence
```




```python
def baseline_score(dataframe,business_id,user_id):
    return dataframe[business_id][user_id]
```


Now, let's split our data into train and test, and use RMSE to evaluate the performance of this approach to calculating baselines.



```python
#Split into test and train
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_total,stratify=df_total['user_id'])
test = test[test['business_id'].isin(train['business_id'])] #This makes sure there is overlap in train and test for both

```




```python
#FORMULA TO RETURN VECTORS OF PREDICTED and ACTUAL VALUES
def baseline_predictions(train,test):
    user_ids = list(set(train['user_id']))
    business_ids = list(set(train['business_id']))
    
    #Getting user and business averages for full matrix    
    business_list = list(set(train['business_id']))
    user_list = list(set(train['user_id']))
    
    business_average_stars = []
    for i in business_list:
        average_stars = np.average(train['stars'], weights=(train['business_id']==i))
        business_average_stars.append(average_stars)
    business_total_avg_stars = np.mean(business_average_stars) #These averages are literally averages of averages - which I think we want
    
    user_average_stars = [] 
    for i in user_list:
        average_stars = np.average(train['stars'], weights=(train['user_id']==i))
        user_average_stars.append(average_stars)
    user_total_avg_stars = np.mean(user_average_stars)
    
    
    
    user_baselines = []
    for i in user_ids:
        a = user_baseline(train,i,user_total_avg_stars)
        user_baselines.append(a)
    
    business_baselines = []
    for i in business_ids:
        a = business_baseline(train,i,business_total_avg_stars)
        business_baselines.append(a)

    #Create matrices of user and business average scores, and then add them
    business_baselines_matrix = np.tile(business_baselines,(len(user_baselines),1))
    user_baselines_matrix = np.tile(user_baselines,(len(business_baselines),1)).transpose()
    
    overall_avg_stars = np.mean(train['stars']) #Perhaps change how this average is calculated
    
    master_baselines_matrix = np.add(business_baselines_matrix,user_baselines_matrix) #Sum the two matrices
    master_baselines_matrix = master_baselines_matrix + overall_avg_stars #Add the average stars from the train dataframe

    #Turn numpy matrix into pandas dataframe with labels for columns and rows
    master_baselines_dataframe = pd.DataFrame(data=master_baselines_matrix,index=user_ids,columns=business_ids)
    
    #Test component: 
    
    #In order to test the accuracy of this, create a dataframe of user-business interactions that actually happened
    test_user_business_combos = list(zip(test['business_id'],
                                         test['user_id'],
                                         test['stars']))
    
    train_user_business_combos = list(zip(train['business_id'],
                                         train['user_id'],
                                         train['stars']))

    train_predicted_values = []
    train_actual_values = []
    for i in train_user_business_combos:
        prediction = baseline_score(master_baselines_dataframe,i[0],i[1])
        #prediction = round(prediction)  ###this line is better off hidden
        train_predicted_values.append(prediction)
        train_actual_values.append(i[2])
    
    train_results = pd.DataFrame({
            'predicted_values': train_predicted_values,
            'actual_values': train_actual_values})
    test_predicted_values = []
    test_actual_values = []
    
    
    for i in test_user_business_combos:
        prediction = baseline_score(master_baselines_dataframe,i[0],i[1])
        #prediction = round(prediction)  ###this line is better off hidden
        test_predicted_values.append(prediction)
        test_actual_values.append(i[2])
    
    test_results = pd.DataFrame({
            'predicted_values': test_predicted_values,
            'actual_values': test_actual_values})

    return test_results,train_results
```




```python
test_results = baseline_predictions(train,test)[0]
```




```python
train_results = baseline_predictions(train,test)[1]
```




```python
#RMSE  
def RMSE(results):
    """
    Calculate the root mean squared error between a matrix of real ratings and predicted ratings
    :param real: A matrix containing the real ratings (with 'NaN' for any missing elements)
    :param predicted: A matrix of predictions
    :return: The RMSE as a float
    """
    return np.sqrt(np.nanmean(np.square(results['actual_values'] - results['predicted_values'])))
```




```python
RMSE(test_results)
```





    1.0654977023908747





```python
RMSE(train_results)
```





    0.88531365781304883





```python
#RIDGE REGRESSION!

#create a formula to remove unecessary columns from a dataframe in order to perform Ridge Regression
def create_ridge_dataframe(df):
    
    #ridge_dataframe = df.drop('text',1)
    #ridge_dataframe = ridge_dataframe.drop('business_id',1)
    #ridge_dataframe = ridge_dataframe.drop('user_id',1)
    ridge_dataframe = ridge_dataframe.drop('review_id',1)
    ridge_dataframe = ridge_dataframe.drop('elite',1)

    ridge_dataframe = ridge_dataframe.drop('user_name',1)
    ridge_dataframe = ridge_dataframe.drop('indexed_id',1)
    ridge_dataframe = ridge_dataframe.drop('address',1)
    ridge_dataframe = ridge_dataframe.drop('attributes',1)
    ridge_dataframe = ridge_dataframe.drop('city',1)
    ridge_dataframe = ridge_dataframe.drop('hours',1)
    ridge_dataframe = ridge_dataframe.drop('is_open',1)
    ridge_dataframe = ridge_dataframe.drop('latitude',1)
    ridge_dataframe = ridge_dataframe.drop('longitude',1)
    ridge_dataframe = ridge_dataframe.drop('business_name',1)
    ridge_dataframe = ridge_dataframe.drop('state',1)
    ridge_dataframe = ridge_dataframe.drop('postal_code',1)
    ridge_dataframe = ridge_dataframe.drop('Restaurant_Status',1)
    ridge_dataframe = ridge_dataframe.drop('categories',1) #Make dummies out of categories, but for now just delete them
    ridge_dataframe = ridge_dataframe.drop('review_date',1) #You should make these usable, but couldn't figure out so far
    ridge_dataframe = ridge_dataframe.drop('yelping_since',1) #Same as above
    ridge_dataframe = ridge_dataframe.drop('user_friends',1)
    ridge_dataframe = ridge_dataframe.drop('business_id',1)
    ridge_dataframe = ridge_dataframe.drop('user_id',1)
    ridge_dataframe = ridge_dataframe.drop('neighborhood',1)
    
    #ridge_dataframe = pd.get_dummies(ridge_dataframe,columns=['neighborhood']) ####DO THIS NEXT
    ridge_dataframe['elite_status'] = ridge_dataframe.elite_status.map(dict(Yes=1, No=0))

    return ridge_dataframe
```




```python
train.columns
```





    Index([u'business_id', u'review_cool', u'review_date', u'review_funny',
           u'review_id', u'stars', u'text', u'review_useful', u'user_id',
           u'compliment_cool', u'compliment_cute', u'compliment_funny',
           u'compliment_hot', u'compliment_list', u'compliment_more',
           u'compliment_note', u'compliment_photos', u'compliment_plain',
           u'compliment_profile', u'compliment_writer', u'user_cool_count',
           u'elite', u'fans', u'user_friends', u'user_funny_count', u'user_name',
           u'user_review_count', u'user_useful_count', u'yelping_since',
           u'elite_status', u'indexed_id', u'address', u'attributes',
           u'categories', u'city', u'hours', u'is_open', u'latitude', u'longitude',
           u'business_name', u'neighborhood', u'postal_code',
           u'business_review_count', u'state', u'Restaurant_Status'],
          dtype='object')





```python
predictors = list(train.columns)
```




```python
def format_ridge(dataframe):
    predictors = list(dataframe.columns)
    predictors.remove('business_id')
    predictors.remove('review_date')
    predictors.remove('review_id')
    predictors.remove('text')
    predictors.remove('user_id')
    predictors.remove('elite')
    predictors.remove('user_friends')
    predictors.remove('user_name')
    predictors.remove('yelping_since')
    predictors.remove('indexed_id')
    predictors.remove('address')
    predictors.remove('attributes')
    predictors.remove('categories')
    predictors.remove('city')
    predictors.remove('hours')
    predictors.remove('is_open')
    predictors.remove('latitude')
    predictors.remove('longitude')
    predictors.remove('business_name')
    predictors.remove('neighborhood')
    predictors.remove('postal_code')
    predictors.remove('state')
    predictors.remove('Restaurant_Status')
    
    new_dataframe = dataframe[predictors]
    new_dataframe['elite_status'] = new_dataframe.elite_status.map(dict(Yes=1, No=0))
    
    return new_dataframe
```




```python
ridge_train = format_ridge(train)
```


    /Users/Kally/anaconda/envs/python2/lib/python2.7/site-packages/ipykernel_launcher.py:28: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy




```python
ridge_train.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_cool</th>
      <th>review_funny</th>
      <th>stars</th>
      <th>review_useful</th>
      <th>compliment_cool</th>
      <th>compliment_cute</th>
      <th>compliment_funny</th>
      <th>compliment_hot</th>
      <th>compliment_list</th>
      <th>compliment_more</th>
      <th>...</th>
      <th>compliment_plain</th>
      <th>compliment_profile</th>
      <th>compliment_writer</th>
      <th>user_cool_count</th>
      <th>fans</th>
      <th>user_funny_count</th>
      <th>user_review_count</th>
      <th>user_useful_count</th>
      <th>elite_status</th>
      <th>business_review_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14974</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>53</td>
    </tr>
    <tr>
      <th>13203</th>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>37551</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>8</td>
      <td>7</td>
      <td>29</td>
      <td>75</td>
      <td>0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>34968</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>26</td>
      <td>24</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>13</td>
      <td>4</td>
      <td>10</td>
      <td>4</td>
      <td>14</td>
      <td>9</td>
      <td>202</td>
      <td>9</td>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>15374</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>118</td>
      <td>2</td>
      <td>0</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>





```python
#Format train and test dataframes accordingly
ridge_train = format_ridge(train)
ridge_test = format_ridge(test)

x_train = add_constant(ridge_train.drop('stars',1))
y_train = ridge_train['stars'].values
y_train = y_train.reshape(len(x_train),1)

x_test = add_constant(ridge_test.drop('stars',1))
y_test = ridge_test['stars'].values
y_test = y_test.reshape(len(x_test),1)
```


    /Users/Kally/anaconda/envs/python2/lib/python2.7/site-packages/ipykernel_launcher.py:28: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy




```python
#Creating lambda list
lambdas = []
for i in range(-5,5):
    lambdas.append(10**i)

#Ridge Regression
from sklearn.model_selection import GridSearchCV
def cv_optimize_ridge(x_train,y_train,lambdas,n_folds=10):
    est = Ridge()
    parameters = {'alpha': lambdas}
    gs = GridSearchCV(est,param_grid=parameters,cv=n_folds,scoring="neg_mean_squared_error")
    gs.fit(x_train,y_train)
    return gs
fitmodel = cv_optimize_ridge(x_train,y_train,lambdas,n_folds = 10)
```




```python
fitmodel.best_params_
```





    {'alpha': 100}





```python
#Running the Ridge regression on the test set
clf = Ridge(alpha= 100 )
clf.fit(x_train, y_train)
clf.predict(x_test)

ridge_preds_test = clf.predict(x_test)
```




```python
#Running the Ridge regression on the train set
clf = Ridge(alpha= 100 )
clf.fit(x_train, y_train)
clf.predict(x_train)

ridge_preds_train = clf.predict(x_train)
```




```python
#RMSE  
def RMSE(actual,predicted):
    """
    Calculate the root mean squared error between a matrix of real ratings and predicted ratings
    :param real: A matrix containing the real ratings (with 'NaN' for any missing elements)
    :param predicted: A matrix of predictions
    :return: The RMSE as a float
    """
    return np.sqrt(np.nanmean(np.square(actual - predicted)))
```




```python
#Calculating the RMSE on the test set
RMSE(y_test,ridge_preds_test)
```





    1.0475227869861707





```python
#Calculating the RMSE on the train set
RMSE(y_train,ridge_preds_train)
```





    1.047741714021146


