---
nav_include: 3
title: Collaborative Filtering
notebook: collaborative_filtering.ipynb
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
from sklearn.metrics import mean_squared_error
from math import sqrt
def cf_rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
```




```python
df_review_train = pd.read_csv("Montreal_review_train.csv")
df_review_test = pd.read_csv("Montreal_review_test.csv")
df_user_train = pd.read_csv("Montreal_user_train.csv")
df_user_test = pd.read_csv("Montreal_user_test.csv")
df_business_train = pd.read_csv("Montreal_business_train.csv")
df_business_test = pd.read_csv("Montreal_business_test.csv")
```




```python
df_user_train.head()
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
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>average_stars</th>
      <th>compliment_cool</th>
      <th>compliment_cute</th>
      <th>compliment_funny</th>
      <th>compliment_hot</th>
      <th>compliment_list</th>
      <th>compliment_more</th>
      <th>compliment_note</th>
      <th>...</th>
      <th>elite</th>
      <th>fans</th>
      <th>friends</th>
      <th>funny</th>
      <th>name</th>
      <th>review_count</th>
      <th>useful</th>
      <th>user_id</th>
      <th>yelping_since</th>
      <th>elite_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10</td>
      <td>4.10</td>
      <td>1302</td>
      <td>41</td>
      <td>1302</td>
      <td>806</td>
      <td>20</td>
      <td>46</td>
      <td>725</td>
      <td>...</td>
      <td>[2016, 2014, 2015, 2017, 2012, 2011, 2013]</td>
      <td>435</td>
      <td>['xRYvFaMGWsvKcLCFtRIzWQ', 'zvQ7B3KZuFOX7pYLsO...</td>
      <td>4880</td>
      <td>Risa</td>
      <td>1122</td>
      <td>26395</td>
      <td>Wc5L6iuvSNF5WGBlqIO8nw</td>
      <td>2011-07-30</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13</td>
      <td>3.79</td>
      <td>1139</td>
      <td>87</td>
      <td>1139</td>
      <td>782</td>
      <td>54</td>
      <td>103</td>
      <td>391</td>
      <td>...</td>
      <td>[2012, 2008, 2009, 2010, 2007, 2006, 2013, 2011]</td>
      <td>198</td>
      <td>['KOwp5RDbm7cDyrdXN8FVQQ', '7MlH7OevWSkenMyKFI...</td>
      <td>10715</td>
      <td>Holly</td>
      <td>698</td>
      <td>24047</td>
      <td>Dd-TkEszFMkSF-vRih51fQ</td>
      <td>2006-07-03</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>37</td>
      <td>3.74</td>
      <td>129</td>
      <td>2</td>
      <td>129</td>
      <td>77</td>
      <td>6</td>
      <td>12</td>
      <td>56</td>
      <td>...</td>
      <td>[2017, 2016, 2012, 2014, 2015, 2011, 2013]</td>
      <td>68</td>
      <td>['Cq8uhBLRO1T9l-9R9OmddQ', 'x3_b9Rv-GZpjtCDLqg...</td>
      <td>105</td>
      <td>Jeff</td>
      <td>754</td>
      <td>151</td>
      <td>YTdNcIWAt2nEzZ7NY-fniw</td>
      <td>2011-05-16</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>110</td>
      <td>4.07</td>
      <td>60</td>
      <td>1</td>
      <td>60</td>
      <td>51</td>
      <td>1</td>
      <td>16</td>
      <td>19</td>
      <td>...</td>
      <td>[2014, 2012, 2015, 2011, 2013, 2016, 2017]</td>
      <td>33</td>
      <td>['8s7UH21vFgkRJAJg2L8VzA', 'HWGrt1MEXlzZ71NGx0...</td>
      <td>9</td>
      <td>Cecille</td>
      <td>356</td>
      <td>36</td>
      <td>bTRFge5pRWMh7IoCLn7lBw</td>
      <td>2007-08-03</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>117</td>
      <td>3.64</td>
      <td>23</td>
      <td>2</td>
      <td>23</td>
      <td>31</td>
      <td>0</td>
      <td>3</td>
      <td>13</td>
      <td>...</td>
      <td>[2012, 2013]</td>
      <td>15</td>
      <td>['G-Hav6XBWPEyzI-0nNpdxw', 'EgqsK7MUgqpbaTVZAv...</td>
      <td>36</td>
      <td>Carolina</td>
      <td>115</td>
      <td>89</td>
      <td>-w7ww3yW5BHE3TFyj3IHuQ</td>
      <td>2010-06-29</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>





```python
print(df_review_train.shape)
print(len(df_review_train['user_id'].unique()))
print(len(df_review_train['business_id'].unique()))

print(df_user_train.shape)
print(len(df_user_train['user_id'].unique()))

print(df_business_train.shape)
print(len(df_business_train['business_id'].unique()))
```


    (30999, 11)
    3201
    2429
    (3201, 25)
    3201
    (2429, 18)
    2429




```python
df_review_train.shape
```





    (30999, 11)





```python
pivot_review_train = df_review_train.pivot(index = 'user_id', columns ='business_id', values = 'stars').fillna(0)
pivot_review_test = df_review_test.pivot(index = 'user_id', columns ='business_id', values = 'stars').fillna(0)
```




```python
pivot_review_train.shape
```





    (3201, 2429)





```python
from scipy.sparse.linalg import svds
```




```python
def user_svd_predict(df, df_):
    R = df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    
    U = []
    sigma = []
    Vt = []
    U, sigma, Vt = svds(R_demeaned, k=20)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    average_rating = list(df_['average_stars'])
    average_rating = np.array(average_rating).reshape(-1, 1)
    average_rating = np.repeat(average_rating, all_user_predicted_ratings.shape[1], axis=1)
    all_user_predicted_ratings_total = all_user_predicted_ratings + average_rating
    return (all_user_predicted_ratings_total, all_user_predicted_ratings)
```




```python
def business_svd_predict(df, df_):
    R = df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    
    U = []
    sigma = []
    Vt = []
    U, sigma, Vt = svds(R_demeaned, k=20)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    average_rating = list(df_['stars'])
    average_rating = np.array(average_rating).reshape(-1, 1)
    average_rating = np.repeat(average_rating, all_user_predicted_ratings.shape[1], axis=1)
    all_user_predicted_ratings_total = all_user_predicted_ratings + average_rating
    return (all_user_predicted_ratings_total, all_user_predicted_ratings)
```




```python
all_user_predicted_ratings_total = user_svd_predict(pivot_review_train, df_user_train)[0]
all_user_predicted_ratings_total_test = user_svd_predict(pivot_review_test, df_user_test)[0]
all_business_predicted_ratings_total = business_svd_predict(pivot_review_train.T, df_business_train)[0]
all_business_predicted_ratings_total_test = business_svd_predict(pivot_review_test.T, df_business_test)[0]
```




```python
pivot_review_train.shape
```





    (3201, 2429)





```python
df_user_train.shape
```





    (3201, 25)





```python
user_id_train = df_user_train['user_id']
user_id_test = df_user_test['user_id']

business_id_train = df_business_train['business_id']
business_id_test = df_business_test['business_id']
```




```python
preds_df_train = pd.DataFrame(all_user_predicted_ratings_total, columns=pivot_review_train.columns, index=user_id_train)
preds_df_train.head()
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
      <th>business_id</th>
      <th>-0uEqc2vw1xXtuI_r1xTNg</th>
      <th>-1xuC540Nycht_iWFeJ-dw</th>
      <th>-7bRnaHp7OHz8KW-THqP4w</th>
      <th>-92cC6-X87HQ1DE1UHOx3w</th>
      <th>-AgfhwHOYrsPKt-_xV_Ipg</th>
      <th>-BPHhtX6zzI59IX7ZY-AQA</th>
      <th>-FDkvLmwaBrtVgYFqEWeWA</th>
      <th>-FPc3kwUU9GTDd4LzurvTQ</th>
      <th>-GHqz1jGYzAtn27CeHeWeA</th>
      <th>-HsqnPAz374YSoyFDyjl3A</th>
      <th>...</th>
      <th>zqV3T9HltH1pmlRFJJSFcA</th>
      <th>zr2wA55AskfBJxrvUeDZRA</th>
      <th>zrnP9HqoF-RI9jqoW8pytA</th>
      <th>zsMMlOYtXm8SNy0bl1leBA</th>
      <th>zsbsLCO-bw3gdNE9XNgBYw</th>
      <th>zv92BYJH09YjFQOtSyYp-A</th>
      <th>zwBEMcCVqh8wOXn_sOIfxg</th>
      <th>zwgVuZcMgijt9k3Jq-2zQQ</th>
      <th>zwkif4XLEDqdEwEgTWLIVQ</th>
      <th>zzjKekzQ6i4iR-qpo405Pw</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Wc5L6iuvSNF5WGBlqIO8nw</th>
      <td>4.111058</td>
      <td>3.765381</td>
      <td>4.079379</td>
      <td>4.128996</td>
      <td>4.074044</td>
      <td>4.113905</td>
      <td>4.090428</td>
      <td>4.105445</td>
      <td>4.112865</td>
      <td>4.236941</td>
      <td>...</td>
      <td>4.072490</td>
      <td>4.138484</td>
      <td>4.098128</td>
      <td>4.177040</td>
      <td>4.119881</td>
      <td>4.092555</td>
      <td>4.095933</td>
      <td>4.094899</td>
      <td>4.219485</td>
      <td>4.085823</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>3.792130</td>
      <td>3.907054</td>
      <td>3.789752</td>
      <td>3.762797</td>
      <td>3.788838</td>
      <td>3.810440</td>
      <td>3.798067</td>
      <td>3.790774</td>
      <td>3.802109</td>
      <td>3.839521</td>
      <td>...</td>
      <td>3.788810</td>
      <td>3.806899</td>
      <td>3.794853</td>
      <td>3.772085</td>
      <td>3.803739</td>
      <td>3.794808</td>
      <td>3.817778</td>
      <td>3.795181</td>
      <td>3.762455</td>
      <td>3.772413</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>3.750744</td>
      <td>3.878034</td>
      <td>3.753158</td>
      <td>3.769249</td>
      <td>3.752987</td>
      <td>3.731422</td>
      <td>3.747570</td>
      <td>3.749814</td>
      <td>3.742257</td>
      <td>3.729388</td>
      <td>...</td>
      <td>3.763453</td>
      <td>3.765689</td>
      <td>3.747911</td>
      <td>3.749756</td>
      <td>3.746374</td>
      <td>3.748165</td>
      <td>3.732125</td>
      <td>3.749004</td>
      <td>3.779889</td>
      <td>3.762268</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>4.115649</td>
      <td>4.236376</td>
      <td>4.113031</td>
      <td>4.187895</td>
      <td>4.109432</td>
      <td>4.201939</td>
      <td>4.100459</td>
      <td>4.078761</td>
      <td>4.103681</td>
      <td>4.439442</td>
      <td>...</td>
      <td>4.131304</td>
      <td>4.116208</td>
      <td>4.090716</td>
      <td>4.198155</td>
      <td>4.069027</td>
      <td>4.081443</td>
      <td>4.195563</td>
      <td>4.100826</td>
      <td>3.985273</td>
      <td>4.088402</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>3.639676</td>
      <td>3.640164</td>
      <td>3.643093</td>
      <td>3.701733</td>
      <td>3.621803</td>
      <td>3.658139</td>
      <td>3.634003</td>
      <td>3.638492</td>
      <td>3.636034</td>
      <td>3.699485</td>
      <td>...</td>
      <td>3.646045</td>
      <td>3.635107</td>
      <td>3.640433</td>
      <td>3.654267</td>
      <td>3.620071</td>
      <td>3.637269</td>
      <td>3.671494</td>
      <td>3.637200</td>
      <td>3.682764</td>
      <td>3.668489</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2429 columns</p>
</div>





```python
preds_df_test = pd.DataFrame(all_user_predicted_ratings_total_test, columns=pivot_review_test.columns, index=user_id_test)
preds_df_test.head()
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
      <th>business_id</th>
      <th>-0uEqc2vw1xXtuI_r1xTNg</th>
      <th>-1xuC540Nycht_iWFeJ-dw</th>
      <th>-7bRnaHp7OHz8KW-THqP4w</th>
      <th>-92cC6-X87HQ1DE1UHOx3w</th>
      <th>-AgfhwHOYrsPKt-_xV_Ipg</th>
      <th>-FDkvLmwaBrtVgYFqEWeWA</th>
      <th>-FPc3kwUU9GTDd4LzurvTQ</th>
      <th>-HsqnPAz374YSoyFDyjl3A</th>
      <th>-MwaICRwxaUi0JBfad2Y3Q</th>
      <th>-Mz3M0g6iFZczs6a7ddf5g</th>
      <th>...</th>
      <th>zktCQRlDtF6XmOpqKBz1mA</th>
      <th>zmQyE-gIUpwBCMmTFFRbJw</th>
      <th>zpw5S3QwUse1MH-Eerbnaw</th>
      <th>zqV3T9HltH1pmlRFJJSFcA</th>
      <th>zr2wA55AskfBJxrvUeDZRA</th>
      <th>zrnP9HqoF-RI9jqoW8pytA</th>
      <th>zsMMlOYtXm8SNy0bl1leBA</th>
      <th>zwBEMcCVqh8wOXn_sOIfxg</th>
      <th>zwgVuZcMgijt9k3Jq-2zQQ</th>
      <th>zwkif4XLEDqdEwEgTWLIVQ</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Wc5L6iuvSNF5WGBlqIO8nw</th>
      <td>4.102477</td>
      <td>4.249559</td>
      <td>4.102253</td>
      <td>4.101849</td>
      <td>4.100737</td>
      <td>4.102513</td>
      <td>4.102370</td>
      <td>4.102075</td>
      <td>4.101559</td>
      <td>4.102597</td>
      <td>...</td>
      <td>4.100987</td>
      <td>4.100863</td>
      <td>4.102219</td>
      <td>4.098799</td>
      <td>4.105492</td>
      <td>4.101588</td>
      <td>4.101210</td>
      <td>4.098801</td>
      <td>4.102365</td>
      <td>4.101979</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>3.790618</td>
      <td>3.788917</td>
      <td>3.790619</td>
      <td>3.790514</td>
      <td>3.790607</td>
      <td>3.790589</td>
      <td>3.790622</td>
      <td>3.790513</td>
      <td>3.790574</td>
      <td>3.790484</td>
      <td>...</td>
      <td>3.790574</td>
      <td>3.790609</td>
      <td>3.790513</td>
      <td>3.790247</td>
      <td>3.790299</td>
      <td>3.790607</td>
      <td>3.790486</td>
      <td>3.790552</td>
      <td>3.790628</td>
      <td>3.790606</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>3.742174</td>
      <td>3.718317</td>
      <td>3.742233</td>
      <td>3.742351</td>
      <td>3.742031</td>
      <td>3.742384</td>
      <td>3.742243</td>
      <td>3.740605</td>
      <td>3.741719</td>
      <td>3.742942</td>
      <td>...</td>
      <td>3.740200</td>
      <td>3.742232</td>
      <td>3.743492</td>
      <td>3.739589</td>
      <td>3.745336</td>
      <td>3.742487</td>
      <td>3.742482</td>
      <td>3.741877</td>
      <td>3.742250</td>
      <td>3.742319</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>4.077589</td>
      <td>4.333084</td>
      <td>4.075519</td>
      <td>4.064777</td>
      <td>4.078651</td>
      <td>4.082149</td>
      <td>4.076959</td>
      <td>4.093477</td>
      <td>4.095684</td>
      <td>4.084624</td>
      <td>...</td>
      <td>4.072748</td>
      <td>4.078266</td>
      <td>4.075081</td>
      <td>4.121961</td>
      <td>4.046767</td>
      <td>4.065211</td>
      <td>4.068705</td>
      <td>4.088010</td>
      <td>4.077159</td>
      <td>4.076365</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>3.642578</td>
      <td>3.688117</td>
      <td>3.642397</td>
      <td>3.645565</td>
      <td>3.641951</td>
      <td>3.642059</td>
      <td>3.642840</td>
      <td>3.642970</td>
      <td>3.649755</td>
      <td>3.640657</td>
      <td>...</td>
      <td>3.644327</td>
      <td>3.643082</td>
      <td>3.643405</td>
      <td>3.664063</td>
      <td>3.646322</td>
      <td>3.641457</td>
      <td>3.644397</td>
      <td>3.642431</td>
      <td>3.642516</td>
      <td>3.642415</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2070 columns</p>
</div>





```python
def simple_recommend(user_id, count):
    df = preds_df_train.T[user_id].sort_values(ascending=False)
    return df.head(count)
```




```python
simple_recommend('yML2P1evj7FrLncIgaFzHw', 10)
```





    business_id
    kKY726bQREexYHHNLK1H7g    4.632830
    IRIlwpomRvnXvpkeaGaM2A    4.390620
    mm2wLW24ESxNIEL2bjseaQ    4.201337
    um_o0pxQ3DlRI9EfCzw0hw    4.194907
    2gUbgbdJ7IFSbicBXlSchw    4.142699
    y32M2Hkr7GsUqGG6KwOhZw    4.079658
    58APdML-PG_OD4El2ePTvw    4.057295
    FhgAHo-8--equM8w5UZ41Q    4.016346
    JN8s_dgw9nrSzkHnXxNOtg    3.909097
    s2I_Ni76bjJNK9yG60iD-Q    3.902061
    Name: yML2P1evj7FrLncIgaFzHw, dtype: float64





```python
from sklearn.metrics.pairwise import pairwise_distances
```




```python
user_similarity_train = 1 - pairwise_distances(user_svd_predict(pivot_review_train, df_user_train)[1], metric='cosine')
user_similarity_test = 1 - pairwise_distances(user_svd_predict(pivot_review_test, df_user_test)[1], metric='cosine')
```




```python
business_similarity_train = 1 - pairwise_distances(business_svd_predict(pivot_review_train.T, df_business_train)[1], metric='cosine')
business_similarity_test = 1 - pairwise_distances(business_svd_predict(pivot_review_test.T, df_business_test)[1], metric='cosine')
```




```python
user_similarity_matrix_train = pd.DataFrame(user_similarity_train, columns=user_id_train, index=user_id_train)
user_similarity_matrix_train.head()
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
      <th>user_id</th>
      <th>Wc5L6iuvSNF5WGBlqIO8nw</th>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <th>4hAauH0dy57uK9o8bCvGUw</th>
      <th>VMfwMYh8iJapW807Pu1Diw</th>
      <th>lKRbcLWDQmOmhcMa3vMCMA</th>
      <th>2vJ2e51kdbdAmAo_HTr4KQ</th>
      <th>9KpMzih4E_gEioFtNeuIIw</th>
      <th>...</th>
      <th>v7q2D8s1vsglwQaQcyb8_A</th>
      <th>hOYNnE3qzb8TDKd3jqvq7Q</th>
      <th>LqywrHdM-H8gSdKtGrhBuw</th>
      <th>iIIbkFd_kgK3n2ewvLstXA</th>
      <th>KJIS0INMJKhBmGqFkHMc-A</th>
      <th>Ih3dwaCS1snsbhS8vRdxHA</th>
      <th>LY-KaOJyXzbwZyqjQfl7xA</th>
      <th>e3XuTKzX3w8LP-mEqQgJ9g</th>
      <th>awdAcl2dA_WvUPWKOCS1OA</th>
      <th>0wXvG8Jiu8zdZhvezBgOwA</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Wc5L6iuvSNF5WGBlqIO8nw</th>
      <td>1.000000</td>
      <td>0.123739</td>
      <td>-0.033120</td>
      <td>0.288995</td>
      <td>0.581747</td>
      <td>0.137748</td>
      <td>0.093395</td>
      <td>0.367794</td>
      <td>0.107102</td>
      <td>-0.062889</td>
      <td>...</td>
      <td>0.317822</td>
      <td>0.267291</td>
      <td>0.089310</td>
      <td>0.074979</td>
      <td>0.187363</td>
      <td>-0.049799</td>
      <td>0.018654</td>
      <td>0.126737</td>
      <td>0.120642</td>
      <td>0.096226</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>0.123739</td>
      <td>1.000000</td>
      <td>0.082072</td>
      <td>0.350151</td>
      <td>0.107605</td>
      <td>0.043687</td>
      <td>0.261767</td>
      <td>-0.083328</td>
      <td>-0.021795</td>
      <td>0.186220</td>
      <td>...</td>
      <td>0.108294</td>
      <td>0.242365</td>
      <td>0.253467</td>
      <td>0.053828</td>
      <td>0.244121</td>
      <td>-0.036538</td>
      <td>0.265373</td>
      <td>0.397636</td>
      <td>0.114079</td>
      <td>-0.098947</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>-0.033120</td>
      <td>0.082072</td>
      <td>1.000000</td>
      <td>0.057411</td>
      <td>0.078848</td>
      <td>0.348380</td>
      <td>0.474439</td>
      <td>0.275562</td>
      <td>0.461942</td>
      <td>0.127301</td>
      <td>...</td>
      <td>-0.010050</td>
      <td>0.215982</td>
      <td>-0.109354</td>
      <td>0.237667</td>
      <td>0.423541</td>
      <td>0.186048</td>
      <td>0.425421</td>
      <td>0.331947</td>
      <td>0.043822</td>
      <td>0.423622</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>0.288995</td>
      <td>0.350151</td>
      <td>0.057411</td>
      <td>1.000000</td>
      <td>0.378264</td>
      <td>0.021736</td>
      <td>0.323239</td>
      <td>0.408859</td>
      <td>0.279707</td>
      <td>0.094915</td>
      <td>...</td>
      <td>0.140430</td>
      <td>-0.173267</td>
      <td>0.514292</td>
      <td>0.148501</td>
      <td>0.373318</td>
      <td>0.159174</td>
      <td>0.036335</td>
      <td>0.105005</td>
      <td>0.271699</td>
      <td>0.127442</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>0.581747</td>
      <td>0.107605</td>
      <td>0.078848</td>
      <td>0.378264</td>
      <td>1.000000</td>
      <td>-0.207281</td>
      <td>0.115763</td>
      <td>0.133768</td>
      <td>0.536527</td>
      <td>0.002525</td>
      <td>...</td>
      <td>0.090438</td>
      <td>-0.006708</td>
      <td>0.176181</td>
      <td>-0.010482</td>
      <td>0.351390</td>
      <td>-0.071426</td>
      <td>0.141043</td>
      <td>0.097905</td>
      <td>0.579939</td>
      <td>-0.221361</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3201 columns</p>
</div>





```python
user_similarity_matrix_test = pd.DataFrame(user_similarity_test, columns=user_id_test, index=user_id_test)
user_similarity_matrix_test.head()
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
      <th>user_id</th>
      <th>Wc5L6iuvSNF5WGBlqIO8nw</th>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <th>4hAauH0dy57uK9o8bCvGUw</th>
      <th>VMfwMYh8iJapW807Pu1Diw</th>
      <th>lKRbcLWDQmOmhcMa3vMCMA</th>
      <th>2vJ2e51kdbdAmAo_HTr4KQ</th>
      <th>9KpMzih4E_gEioFtNeuIIw</th>
      <th>...</th>
      <th>v7q2D8s1vsglwQaQcyb8_A</th>
      <th>hOYNnE3qzb8TDKd3jqvq7Q</th>
      <th>LqywrHdM-H8gSdKtGrhBuw</th>
      <th>iIIbkFd_kgK3n2ewvLstXA</th>
      <th>KJIS0INMJKhBmGqFkHMc-A</th>
      <th>Ih3dwaCS1snsbhS8vRdxHA</th>
      <th>LY-KaOJyXzbwZyqjQfl7xA</th>
      <th>e3XuTKzX3w8LP-mEqQgJ9g</th>
      <th>awdAcl2dA_WvUPWKOCS1OA</th>
      <th>0wXvG8Jiu8zdZhvezBgOwA</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Wc5L6iuvSNF5WGBlqIO8nw</th>
      <td>1.000000</td>
      <td>0.273028</td>
      <td>0.029052</td>
      <td>0.184856</td>
      <td>0.024051</td>
      <td>0.461267</td>
      <td>0.180832</td>
      <td>-0.215365</td>
      <td>-0.025200</td>
      <td>0.160955</td>
      <td>...</td>
      <td>0.234125</td>
      <td>-0.125812</td>
      <td>0.262517</td>
      <td>0.101626</td>
      <td>0.191844</td>
      <td>0.252877</td>
      <td>-0.068769</td>
      <td>0.252614</td>
      <td>-0.127824</td>
      <td>0.112969</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>0.273028</td>
      <td>1.000000</td>
      <td>0.756588</td>
      <td>0.152833</td>
      <td>0.264246</td>
      <td>0.858565</td>
      <td>0.570801</td>
      <td>0.283083</td>
      <td>-0.025041</td>
      <td>0.529752</td>
      <td>...</td>
      <td>0.934899</td>
      <td>0.572209</td>
      <td>0.580154</td>
      <td>0.787125</td>
      <td>0.327144</td>
      <td>-0.019333</td>
      <td>0.120580</td>
      <td>0.962273</td>
      <td>0.103892</td>
      <td>0.289312</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>0.029052</td>
      <td>0.756588</td>
      <td>1.000000</td>
      <td>0.144151</td>
      <td>0.128624</td>
      <td>0.711422</td>
      <td>0.576731</td>
      <td>0.552788</td>
      <td>-0.020392</td>
      <td>0.799825</td>
      <td>...</td>
      <td>0.787761</td>
      <td>0.504840</td>
      <td>0.515459</td>
      <td>0.643395</td>
      <td>0.443141</td>
      <td>0.199837</td>
      <td>0.492037</td>
      <td>0.668754</td>
      <td>0.193777</td>
      <td>0.227147</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>0.184856</td>
      <td>0.152833</td>
      <td>0.144151</td>
      <td>1.000000</td>
      <td>0.342405</td>
      <td>0.138795</td>
      <td>0.421830</td>
      <td>0.194041</td>
      <td>0.101054</td>
      <td>0.136212</td>
      <td>...</td>
      <td>0.246640</td>
      <td>0.014915</td>
      <td>0.265498</td>
      <td>0.147580</td>
      <td>0.369932</td>
      <td>0.225324</td>
      <td>-0.008432</td>
      <td>0.081523</td>
      <td>-0.241318</td>
      <td>-0.066618</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>0.024051</td>
      <td>0.264246</td>
      <td>0.128624</td>
      <td>0.342405</td>
      <td>1.000000</td>
      <td>0.218998</td>
      <td>0.527180</td>
      <td>-0.051809</td>
      <td>-0.016700</td>
      <td>0.247150</td>
      <td>...</td>
      <td>0.235919</td>
      <td>0.186573</td>
      <td>0.156075</td>
      <td>0.305037</td>
      <td>0.159912</td>
      <td>-0.208084</td>
      <td>-0.007545</td>
      <td>0.276780</td>
      <td>0.130077</td>
      <td>0.028004</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2800 columns</p>
</div>





```python
business_similarity_matrix_train = pd.DataFrame(business_similarity_train, columns=business_id_train, index=business_id_train)
business_similarity_matrix_train.head()
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
      <th>business_id</th>
      <th>58APdML-PG_OD4El2ePTvw</th>
      <th>8Rdz0VPY8CuT2GQZ7ho2sw</th>
      <th>DAMTCTsSeACXbkSABkhZqQ</th>
      <th>6I6uDGwCDggrWXi2T4lfaA</th>
      <th>qUdGBSFkiPhEL6I718y-Gg</th>
      <th>ujcbqs6jZfaESgSLvbjWuQ</th>
      <th>XjbPr3o-YTsticeavLjTEg</th>
      <th>Y22IfhXChXoRp3vKi6QwaQ</th>
      <th>MhINNBBwzGn4-n_YI67wog</th>
      <th>OLg1IeS-QxZgNprQ4Hg9gg</th>
      <th>...</th>
      <th>LLBmqBunk40IHdHH_QfjkA</th>
      <th>-ZHeHh4bwLlecbcAD7fTqw</th>
      <th>SnD7fcwR4NR7Cgtx7Qm4ZQ</th>
      <th>ml7HQlaAcszdBZZHljvYgg</th>
      <th>Y5I-z2S3Eeno6cDyn0e6Cg</th>
      <th>ODZLMTbjCnpDNkW1JbMjlQ</th>
      <th>kWDAdT4m3vbnmE0CgLs4gA</th>
      <th>rofWaZTIuaedAxT_UKleSw</th>
      <th>bYfEp3NMskYfEzWL8tVb4w</th>
      <th>HzUxQ1WpeNmeecXN-HPlPw</th>
    </tr>
    <tr>
      <th>business_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58APdML-PG_OD4El2ePTvw</th>
      <td>1.000000</td>
      <td>0.271844</td>
      <td>0.079825</td>
      <td>0.428932</td>
      <td>0.314548</td>
      <td>0.304685</td>
      <td>0.176533</td>
      <td>0.329332</td>
      <td>0.320960</td>
      <td>0.537847</td>
      <td>...</td>
      <td>0.283587</td>
      <td>0.349941</td>
      <td>0.018316</td>
      <td>0.433927</td>
      <td>0.185373</td>
      <td>0.038287</td>
      <td>-0.286370</td>
      <td>-0.044779</td>
      <td>0.350037</td>
      <td>0.162044</td>
    </tr>
    <tr>
      <th>8Rdz0VPY8CuT2GQZ7ho2sw</th>
      <td>0.271844</td>
      <td>1.000000</td>
      <td>-0.060567</td>
      <td>0.191821</td>
      <td>0.296899</td>
      <td>0.099059</td>
      <td>-0.033960</td>
      <td>0.052488</td>
      <td>0.032706</td>
      <td>0.248680</td>
      <td>...</td>
      <td>0.210382</td>
      <td>0.143372</td>
      <td>-0.075271</td>
      <td>0.367880</td>
      <td>0.214407</td>
      <td>0.050381</td>
      <td>-0.001927</td>
      <td>0.117788</td>
      <td>0.357570</td>
      <td>0.531283</td>
    </tr>
    <tr>
      <th>DAMTCTsSeACXbkSABkhZqQ</th>
      <td>0.079825</td>
      <td>-0.060567</td>
      <td>1.000000</td>
      <td>0.344558</td>
      <td>0.041525</td>
      <td>0.080128</td>
      <td>-0.008973</td>
      <td>-0.141186</td>
      <td>-0.304736</td>
      <td>0.151596</td>
      <td>...</td>
      <td>0.302561</td>
      <td>0.119427</td>
      <td>0.007250</td>
      <td>-0.082208</td>
      <td>-0.231451</td>
      <td>-0.235235</td>
      <td>0.129771</td>
      <td>0.020775</td>
      <td>-0.147551</td>
      <td>0.295134</td>
    </tr>
    <tr>
      <th>6I6uDGwCDggrWXi2T4lfaA</th>
      <td>0.428932</td>
      <td>0.191821</td>
      <td>0.344558</td>
      <td>1.000000</td>
      <td>0.566044</td>
      <td>0.147380</td>
      <td>0.233676</td>
      <td>0.181992</td>
      <td>0.093687</td>
      <td>0.269127</td>
      <td>...</td>
      <td>0.281470</td>
      <td>0.342534</td>
      <td>0.178220</td>
      <td>0.417244</td>
      <td>-0.037037</td>
      <td>-0.043609</td>
      <td>0.100768</td>
      <td>-0.097680</td>
      <td>0.602414</td>
      <td>0.637140</td>
    </tr>
    <tr>
      <th>qUdGBSFkiPhEL6I718y-Gg</th>
      <td>0.314548</td>
      <td>0.296899</td>
      <td>0.041525</td>
      <td>0.566044</td>
      <td>1.000000</td>
      <td>-0.030378</td>
      <td>0.244858</td>
      <td>0.121635</td>
      <td>0.357527</td>
      <td>0.103237</td>
      <td>...</td>
      <td>0.363627</td>
      <td>0.323686</td>
      <td>0.165245</td>
      <td>0.420615</td>
      <td>-0.010195</td>
      <td>-0.153927</td>
      <td>-0.098615</td>
      <td>0.079447</td>
      <td>0.414936</td>
      <td>0.419869</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2429 columns</p>
</div>





```python
business_similarity_matrix_test = pd.DataFrame(business_similarity_test, columns=business_id_test, index=business_id_test)
business_similarity_matrix_test.head()
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
      <th>business_id</th>
      <th>58APdML-PG_OD4El2ePTvw</th>
      <th>DAMTCTsSeACXbkSABkhZqQ</th>
      <th>6I6uDGwCDggrWXi2T4lfaA</th>
      <th>qUdGBSFkiPhEL6I718y-Gg</th>
      <th>ujcbqs6jZfaESgSLvbjWuQ</th>
      <th>Y22IfhXChXoRp3vKi6QwaQ</th>
      <th>MhINNBBwzGn4-n_YI67wog</th>
      <th>OLg1IeS-QxZgNprQ4Hg9gg</th>
      <th>DwJlGxAJvohbDR_5jV-ERA</th>
      <th>i5j3FrxdR224KIjfv8x2CQ</th>
      <th>...</th>
      <th>3uu5jvP5JKdSUW9jk-HO7A</th>
      <th>Akhq4AKxKRDPa6BHpiSEVQ</th>
      <th>LLBmqBunk40IHdHH_QfjkA</th>
      <th>-ZHeHh4bwLlecbcAD7fTqw</th>
      <th>SnD7fcwR4NR7Cgtx7Qm4ZQ</th>
      <th>ml7HQlaAcszdBZZHljvYgg</th>
      <th>Y5I-z2S3Eeno6cDyn0e6Cg</th>
      <th>rofWaZTIuaedAxT_UKleSw</th>
      <th>bYfEp3NMskYfEzWL8tVb4w</th>
      <th>HzUxQ1WpeNmeecXN-HPlPw</th>
    </tr>
    <tr>
      <th>business_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58APdML-PG_OD4El2ePTvw</th>
      <td>1.000000</td>
      <td>-0.023429</td>
      <td>0.234868</td>
      <td>0.231090</td>
      <td>0.091581</td>
      <td>0.490803</td>
      <td>0.710535</td>
      <td>0.365032</td>
      <td>0.206606</td>
      <td>0.239764</td>
      <td>...</td>
      <td>-0.105276</td>
      <td>0.373045</td>
      <td>0.028260</td>
      <td>0.139765</td>
      <td>0.027663</td>
      <td>0.213343</td>
      <td>0.297128</td>
      <td>0.192612</td>
      <td>0.847896</td>
      <td>0.792960</td>
    </tr>
    <tr>
      <th>DAMTCTsSeACXbkSABkhZqQ</th>
      <td>-0.023429</td>
      <td>1.000000</td>
      <td>0.013408</td>
      <td>-0.056256</td>
      <td>0.099845</td>
      <td>-0.113187</td>
      <td>-0.060088</td>
      <td>0.582035</td>
      <td>0.032511</td>
      <td>-0.043455</td>
      <td>...</td>
      <td>-0.017552</td>
      <td>-0.115553</td>
      <td>0.365847</td>
      <td>-0.009046</td>
      <td>0.496379</td>
      <td>0.022771</td>
      <td>-0.121338</td>
      <td>-0.001104</td>
      <td>-0.041568</td>
      <td>-0.024040</td>
    </tr>
    <tr>
      <th>6I6uDGwCDggrWXi2T4lfaA</th>
      <td>0.234868</td>
      <td>0.013408</td>
      <td>1.000000</td>
      <td>0.023901</td>
      <td>-0.007454</td>
      <td>0.240044</td>
      <td>0.315998</td>
      <td>0.516830</td>
      <td>-0.185376</td>
      <td>0.057671</td>
      <td>...</td>
      <td>-0.017692</td>
      <td>0.178879</td>
      <td>0.080482</td>
      <td>-0.038698</td>
      <td>0.006968</td>
      <td>0.828757</td>
      <td>0.000083</td>
      <td>-0.022214</td>
      <td>0.342843</td>
      <td>0.480901</td>
    </tr>
    <tr>
      <th>qUdGBSFkiPhEL6I718y-Gg</th>
      <td>0.231090</td>
      <td>-0.056256</td>
      <td>0.023901</td>
      <td>1.000000</td>
      <td>0.036277</td>
      <td>0.230125</td>
      <td>0.330451</td>
      <td>0.077361</td>
      <td>0.199359</td>
      <td>0.448048</td>
      <td>...</td>
      <td>-0.061041</td>
      <td>0.359701</td>
      <td>0.173053</td>
      <td>0.042803</td>
      <td>0.300711</td>
      <td>0.026037</td>
      <td>0.238533</td>
      <td>-0.153242</td>
      <td>0.378002</td>
      <td>0.386268</td>
    </tr>
    <tr>
      <th>ujcbqs6jZfaESgSLvbjWuQ</th>
      <td>0.091581</td>
      <td>0.099845</td>
      <td>-0.007454</td>
      <td>0.036277</td>
      <td>1.000000</td>
      <td>0.085431</td>
      <td>0.067322</td>
      <td>0.350011</td>
      <td>-0.409916</td>
      <td>0.101035</td>
      <td>...</td>
      <td>0.103053</td>
      <td>0.330126</td>
      <td>0.238725</td>
      <td>0.149777</td>
      <td>-0.080113</td>
      <td>-0.096041</td>
      <td>0.187941</td>
      <td>-0.123287</td>
      <td>0.186262</td>
      <td>0.152023</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2070 columns</p>
</div>





```python
df1 = pd.merge(df_user_train, df_review_train, on='user_id')
df_train_total = pd.merge(df1, df_business_train, on='business_id')

df2 = pd.merge(df_user_test, df_review_test, on='user_id')
df_test_total = pd.merge(df2, df_business_test, on='business_id')
```




```python
pivot_user_train = df_train_total.pivot(index = 'user_id', columns ='business_id', values = 'average_stars').fillna(0)
pivot_user_test = df_test_total.pivot(index = 'user_id', columns ='business_id', values = 'average_stars').fillna(0)
pivot_business_train = df_train_total.pivot(index = 'user_id', columns ='business_id', values = 'stars_y').fillna(0)
pivot_business_test = df_test_total.pivot(index = 'user_id', columns ='business_id', values = 'stars_y').fillna(0)
```




```python
items_train = df_review_train.shape[0]
total_train = np.sum(df_review_train['stars'])
global_mean_train = total_train / items_train
print(global_mean_train)
pivot_user_train[pivot_user_train != 0] = global_mean_train

items_test = df_review_test.shape[0]
total_test = np.sum(df_review_test['stars'])
global_mean_test = total_test / items_test
print(global_mean_test)
pivot_user_test[pivot_user_test != 0] = global_mean_test
```


    3.820671634568857
    3.8194511314395765




```python
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
```




```python
train_data_matrix = user_svd_predict(pivot_review_train, df_user_train)[1]
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

test_data_matrix = user_svd_predict(pivot_review_test, df_user_test)[1]
item_similarity_test = pairwise_distances(test_data_matrix.T, metric='cosine')
user_similarity_test = pairwise_distances(test_data_matrix, metric='cosine')
item_prediction_test = predict(test_data_matrix, item_similarity_test, type='item')
user_prediction_test = predict(test_data_matrix, user_similarity_test, type='user')
```




```python
pivot_train = pivot_review_train.as_matrix()
pivot_test = pivot_review_test.as_matrix()
pivot_user_train_ = pivot_user_train.as_matrix()
pivot_user_test_ = pivot_user_test.as_matrix()
pivot_business_train_ = pivot_business_train.as_matrix()
pivot_business_test_ = pivot_business_test.as_matrix()
```




```python
pivot_pred_train = np.add(item_prediction, pivot_user_train_)
pivot_pred_train = np.add(pivot_pred_train, user_prediction)

pivot_pred_test = np.add(item_prediction_test, pivot_user_test_)
pivot_pred_test = np.add(pivot_pred_test, user_prediction_test)
```




```python
print("CF Train RMSE score is {}".format(cf_rmse(pivot_train, pivot_pred_train)))
print("CF Test RMSE score is {}".format(cf_rmse(pivot_test, pivot_pred_test)))
```


    CF Train RMSE score is 0.08932886240815875
    CF Test RMSE score is 0.05339812124201512

