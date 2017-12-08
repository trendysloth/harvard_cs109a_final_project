---
nav_include: 1
title: Ensemble
notebook: ensemble.ipynb
---

### Contents
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

os.chdir('/Users/thomashill/Documents/Education/Fall 2017/Comp Sci/Final Project/Data/dataset/Cities_dfs')
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

train, test = train_test_split(df_total,stratify=df_total['user_id'],random_state=1990)
train = train[train['business_id'].isin(test['business_id'])] #This makes sure there is overlap in train and test for both
test = test[test['business_id'].isin(train['business_id'])]
train = train[train['user_id'].isin(test['user_id'])]
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

    return test_results,train_results,user_baselines_matrix,business_baselines_matrix
```




```python
baseline_predictions_set = baseline_predictions(train,test)
test_results = baseline_predictions_set[0]
```




```python
train_results = baseline_predictions_set[1]
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
#RMSE of the arithmetic baselines on the test set:
RMSE(test_results)
```





    1.0686277753151672





```python
#RMSE of the arithmetic baselines on the train set:
RMSE(train_results)
```





    0.87568246661790661





```python
#Formula for endogenously calculating user and business average stars

def endog_avg(dataframe):
    users_list = list(set(dataframe['user_id']))
    business_list = list(set(dataframe['business_id']))
    
    user_avg_stars_endog = []
    for i in users_list:
        filtered_df = dataframe[dataframe['user_id']==i]
        mean_stars = np.mean(filtered_df['stars'])
        user_avg_stars_endog.append(mean_stars)
    
    user_stars_dict = dict(zip(users_list,user_avg_stars_endog))

    bus_avg_stars_endog = []
    for i in business_list:
        filtered_df = dataframe[dataframe['business_id']==i]
        mean_stars = np.mean(filtered_df['stars'])
        bus_avg_stars_endog.append(mean_stars)
    
    bus_stars_dict = dict(zip(business_list,bus_avg_stars_endog))
    
    #Add endogenously-calculated user and business average stars to the dataframe
    user_avg_stars = []
    for i in dataframe['user_id']:
        user_avg_stars.append(user_stars_dict[i])
    dataframe = dataframe.assign(user_avg_stars = user_avg_stars)

    bus_avg_stars = []
    for i in dataframe['business_id']:
        bus_avg_stars.append(bus_stars_dict[i])
    dataframe = dataframe.assign(bus_avg_stars = bus_avg_stars)
    
    #Add endogenously-calculated user and business average stars to the dataframe
    user_avg_stars = []
    for i in dataframe['user_id']:
        user_avg_stars.append(user_stars_dict[i])
    dataframe = dataframe.assign(user_avg_stars = user_avg_stars)

    bus_avg_stars = []
    for i in dataframe['business_id']:
        bus_avg_stars.append(bus_stars_dict[i])
    dataframe = dataframe.assign(bus_avg_stars = bus_avg_stars)
    
    return dataframe
```




```python
train = endog_avg(train)
test = endog_avg(test)
```




```python
#RIDGE REGRESSION!
```




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


    /Users/thomashill/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:28: SettingWithCopyWarning: 
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
      <th>compliment_writer</th>
      <th>user_cool_count</th>
      <th>fans</th>
      <th>user_funny_count</th>
      <th>user_review_count</th>
      <th>user_useful_count</th>
      <th>elite_status</th>
      <th>business_review_count</th>
      <th>user_avg_stars</th>
      <th>bus_avg_stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26713</th>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>4</td>
      <td>106</td>
      <td>5</td>
      <td>1</td>
      <td>67</td>
      <td>3.600000</td>
      <td>4.233333</td>
    </tr>
    <tr>
      <th>37534</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>56</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>3.538462</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>35661</th>
      <td>22</td>
      <td>18</td>
      <td>5</td>
      <td>24</td>
      <td>2550</td>
      <td>44</td>
      <td>2550</td>
      <td>1547</td>
      <td>1</td>
      <td>190</td>
      <td>...</td>
      <td>376</td>
      <td>17341</td>
      <td>152</td>
      <td>9170</td>
      <td>601</td>
      <td>17822</td>
      <td>1</td>
      <td>6</td>
      <td>3.695652</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>16762</th>
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
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>95</td>
      <td>3.600000</td>
      <td>4.354167</td>
    </tr>
    <tr>
      <th>28147</th>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>19</td>
      <td>2</td>
      <td>3</td>
      <td>16</td>
      <td>13</td>
      <td>0</td>
      <td>10</td>
      <td>4.800000</td>
      <td>4.600000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
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


    /Users/thomashill/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:28: SettingWithCopyWarning: 
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





    {'alpha': 10}





```python
#Running the Ridge regression on the test set
clf = Ridge(alpha= 1 )
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
#Calculating the RMSE on the test set using Ridge
RMSE(y_test,ridge_preds_test)
```





    0.7309934541268529





```python
#Calculating the RMSE on the train set using Ridge
RMSE(y_train,ridge_preds_train)
```





    0.84852178783719645





```python
#SINGULAR VALUE DECOMPOSITION

#SVD - setting up
train_pivot = train.pivot(index = 'user_id', columns ='business_id', values = 'stars').fillna(0)
train_matrix = train_pivot.as_matrix()
train_matrix_mean = np.mean(train_matrix, axis = 1)
train_matrix_demeaned = train_matrix - train_matrix_mean.reshape(-1, 1)

test_pivot = test.pivot(index = 'user_id', columns ='business_id', values = 'stars').fillna(0)
test_matrix = test_pivot.as_matrix()
test_matrix_mean = np.mean(test_matrix, axis = 1)
test_matrix_demeaned = test_matrix - test_matrix_mean.reshape(-1, 1)
```




```python
#Getting baseline averages
user_baselines_matrix = baseline_predictions_set[2]
business_baselines_matrix = baseline_predictions_set[3]

overall_avg_stars = np.mean(train['stars']) #Perhaps change how this average is calculated
    
master_baselines_matrix = np.add(business_baselines_matrix,user_baselines_matrix) #Sum the two matrices
master_baselines_matrix = master_baselines_matrix + overall_avg_stars
```




```python
#Singular Value Decomposition
from scipy.sparse.linalg import svds

U_train, sigma_train, Vt_train = svds(train_matrix_demeaned, k = 50)
sigma_train = np.diag(sigma_train)

U_test, sigma_test, Vt_test = svds(test_matrix_demeaned, k = 50)
sigma_test = np.diag(sigma_test)

train_predicted_ratings = np.dot(np.dot(U_train, sigma_train), Vt_train) + master_baselines_matrix #+ train_matrix_mean.reshape(-1, 1) #replace these with the baseline mean
test_predicted_ratings = np.dot(np.dot(U_test, sigma_test), Vt_test) + master_baselines_matrix #+ test_matrix_mean.reshape(-1, 1) #replace these with the baseline mean

train_preds_df = pd.DataFrame(train_predicted_ratings, columns = train_pivot.columns, index=train_pivot.index)
test_preds_df = pd.DataFrame(test_predicted_ratings, columns = test_pivot.columns,index=test_pivot.index)

#print(test_preds_df.shape)
#test_preds_df.head()
```




```python
train_preds_df.shape
small_df = train_preds_df.sample(n=50, axis=1)
small_df2 = small_df.sample(n=50,axis=0)
print(small_df2.shape)
small_df2.head()
vals=[]
for i in small_df2.columns:
    vals.append(small_df2[i].values.sum())

small_df2['sums'] = vals
small_df2.sort_values(by='sums', inplace=True)
```


    (50, 50)




```python
#Getting predicted values
user_ids = list(set(train['user_id']))
business_ids = list(set(train['business_id']))
    
#Getting user and business averages for full matrix    
business_list = list(set(train['business_id']))
user_list = list(set(train['user_id']))
```




```python
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
    prediction = train_preds_df[i[0]][i[1]]
    #prediction = round(prediction)  ###this line is better off hidden
    train_predicted_values.append(prediction)
    train_actual_values.append(i[2])


newlist = []
for item in train_predicted_values:
    if item > 5:
        item = 5
    newlist.append(item)
train_predicted_values = newlist    
    
train_results_svd = pd.DataFrame({
        'predicted_values': train_predicted_values,
        'actual_values': train_actual_values})

test_predicted_values = []
test_actual_values = []
    
    
for i in test_user_business_combos:
    prediction = test_preds_df[i[0]][i[1]]
    #prediction = round(prediction)  ###this line is better off hidden
    test_predicted_values.append(prediction)
    test_actual_values.append(i[2])

newlist = []
for item in test_predicted_values:
    if item > 5:
        item = 5
    newlist.append(item)
test_predicted_values = newlist 

test_results_svd = pd.DataFrame({
        'predicted_values': test_predicted_values,
        'actual_values': test_actual_values})
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
#RMSE on the test set using SVD
RMSE(test_results_svd['actual_values'],test_results_svd['predicted_values'])
```





    1.3574296321368511





```python
#RMSE on the train set using SVD
RMSE(train_results_svd['actual_values'],train_results_svd['predicted_values'])
```





    1.33190580432705





```python
#Some simple ensembling

#train
x_train_ensemble = pd.DataFrame({
    'ridge_preds': list(ridge_preds_train),
    'arith_preds': list(train_results['predicted_values']),
    'svd_preds': list(train_results_svd['predicted_values'])})

#test
x_test_ensemble = pd.DataFrame({
    'ridge_preds': list(ridge_preds_test),
    'arith_preds': list(test_results['predicted_values']),
    'svd_preds': list(test_results_svd['predicted_values'])})

```




```python
#Ridge method
rm = Ridge(alpha=100.0)
rm.fit(x_train_ensemble, y_train)

train_predicted_scores_Ridge = rm.predict(x_train_ensemble)
test_predicted_scores_Ridge = rm.predict(x_test_ensemble)

#test RMSE of the three approaches ensembled using Ridge
RMSE(y_test,test_predicted_scores_Ridge)
```





    0.72747361527104459



As we can see from the above, the ensemble method with the three approaches stacked into a ridge regression model produces a RMSE on the test set of 0.727, which is the best of all models we were able to find. Some alternative ensemble models are demonstrated below. 



```python
#KNN
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(x_train_ensemble,y_train)
train_predicted_scores_KNN = KNN.predict(x_train_ensemble)
test_predicted_scores_KNN = KNN.predict(x_test_ensemble)

RMSE(y_test,test_predicted_scores_KNN)
```


    /Users/thomashill/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:3: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      This is separate from the ipykernel package so we can avoid doing imports until





    1.5367768203443726





```python
#Lasso
lasso = LassoCV(fit_intercept=False)
lasso.fit(x_train_ensemble,y_train)
train_predicted_scores_Lasso = lasso.predict(x_train_ensemble)
test_predicted_scores_Lasso = lasso.predict(x_test_ensemble)

RMSE(y_test,test_predicted_scores_Lasso)
```


    /Users/thomashill/anaconda/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:1082: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)





    1.3403076216822609





```python
#Linear Regression
linear = LinearRegression(fit_intercept=False)
linear.fit(x_train_ensemble,y_train)
train_predicted_scores_Linear = linear.predict(x_train_ensemble)
test_predicted_scores_Linear = linear.predict(x_test_ensemble)

RMSE(y_test,test_predicted_scores_Linear)
```





    0.72589325736783528





```python
#Meta ensemble:

#train
x_train_ensemble_meta = pd.DataFrame({
    'Ridge': list(train_predicted_scores_Ridge),
    'KNN': list(train_predicted_scores_KNN),
    'Lasso': list(train_predicted_scores_Lasso),
    'Linear': list(train_predicted_scores_Linear)})
        
#test
x_test_ensemble_meta = pd.DataFrame({
    'Ridge': list(test_predicted_scores_Ridge),
    'KNN': list(test_predicted_scores_KNN),
    'Lasso': list(test_predicted_scores_Lasso),
    'Linear': list(test_predicted_scores_Linear)})

```




```python
#Meta Ridge method
rm = Ridge(alpha=100.0)
rm.fit(x_train_ensemble_meta, y_train)

train_predicted_scores_Ridge = rm.predict(x_train_ensemble_meta)
test_predicted_scores_Ridge = rm.predict(x_test_ensemble_meta)

RMSE(y_test,test_predicted_scores_Ridge)
```





    0.74949184344584108


