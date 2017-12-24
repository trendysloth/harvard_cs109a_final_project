---
nav_include: 1
title: Arithmetic Baseline Model
notebook: arithmetic_baseline_model.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}



```python
#Set up workspace
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from collections import Counter
```




```python
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


The above values are the divergence of the individual or business from the average. 



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

    return test_results
```




```python
results = baseline_predictions(train,test)
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
RMSE(results)
```





    1.0643776540737961


