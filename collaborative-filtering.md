---
nav_include: 2
title: Collaborative-filtering
notebook: collaborative-filtering.ipynb
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
pivot_review_train = df_review_train.pivot(index = 'user_id', columns ='business_id', values = 'stars').fillna(0)
pivot_review_test = df_review_test.pivot(index = 'user_id', columns ='business_id', values = 'stars').fillna(0)
```




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
      <td>4.094261</td>
      <td>4.240964</td>
      <td>4.093483</td>
      <td>4.127661</td>
      <td>4.099962</td>
      <td>4.098311</td>
      <td>4.099023</td>
      <td>4.093190</td>
      <td>4.094802</td>
      <td>4.143096</td>
      <td>...</td>
      <td>4.109433</td>
      <td>4.132640</td>
      <td>4.094520</td>
      <td>4.115686</td>
      <td>4.102577</td>
      <td>4.093049</td>
      <td>4.098747</td>
      <td>4.095100</td>
      <td>4.119639</td>
      <td>4.098103</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>3.795136</td>
      <td>3.815809</td>
      <td>3.795515</td>
      <td>3.798954</td>
      <td>3.795709</td>
      <td>3.795789</td>
      <td>3.796298</td>
      <td>3.795232</td>
      <td>3.795024</td>
      <td>3.803154</td>
      <td>...</td>
      <td>3.795918</td>
      <td>3.795367</td>
      <td>3.795871</td>
      <td>3.796287</td>
      <td>3.797605</td>
      <td>3.795475</td>
      <td>3.797292</td>
      <td>3.795636</td>
      <td>3.797662</td>
      <td>3.795856</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>3.749236</td>
      <td>3.817079</td>
      <td>3.748527</td>
      <td>3.757432</td>
      <td>3.750475</td>
      <td>3.751114</td>
      <td>3.749137</td>
      <td>3.748381</td>
      <td>3.748697</td>
      <td>3.757516</td>
      <td>...</td>
      <td>3.756116</td>
      <td>3.770139</td>
      <td>3.747204</td>
      <td>3.758025</td>
      <td>3.747736</td>
      <td>3.747685</td>
      <td>3.747688</td>
      <td>3.748463</td>
      <td>3.756472</td>
      <td>3.751034</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>4.092210</td>
      <td>4.509308</td>
      <td>4.090106</td>
      <td>4.152235</td>
      <td>4.101792</td>
      <td>4.104338</td>
      <td>4.097496</td>
      <td>4.088443</td>
      <td>4.089012</td>
      <td>4.167343</td>
      <td>...</td>
      <td>4.132688</td>
      <td>4.203549</td>
      <td>4.084595</td>
      <td>4.142995</td>
      <td>4.095679</td>
      <td>4.085720</td>
      <td>4.093583</td>
      <td>4.090215</td>
      <td>4.140908</td>
      <td>4.104515</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>3.639581</td>
      <td>3.764347</td>
      <td>3.639625</td>
      <td>3.662383</td>
      <td>3.638872</td>
      <td>3.644714</td>
      <td>3.642259</td>
      <td>3.638670</td>
      <td>3.636508</td>
      <td>3.672317</td>
      <td>...</td>
      <td>3.651202</td>
      <td>3.673610</td>
      <td>3.638274</td>
      <td>3.655888</td>
      <td>3.643778</td>
      <td>3.638415</td>
      <td>3.641998</td>
      <td>3.639788</td>
      <td>3.654747</td>
      <td>3.642414</td>
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
      <td>4.102510</td>
      <td>4.110007</td>
      <td>4.102534</td>
      <td>4.102885</td>
      <td>4.102467</td>
      <td>4.102479</td>
      <td>4.102763</td>
      <td>4.102247</td>
      <td>4.102289</td>
      <td>4.104428</td>
      <td>...</td>
      <td>4.103460</td>
      <td>4.102425</td>
      <td>4.102717</td>
      <td>4.099520</td>
      <td>4.103339</td>
      <td>4.102584</td>
      <td>4.102180</td>
      <td>4.102489</td>
      <td>4.102553</td>
      <td>4.102519</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>3.790563</td>
      <td>3.789336</td>
      <td>3.790566</td>
      <td>3.790516</td>
      <td>3.790554</td>
      <td>3.790560</td>
      <td>3.790560</td>
      <td>3.790465</td>
      <td>3.790527</td>
      <td>3.790369</td>
      <td>...</td>
      <td>3.790459</td>
      <td>3.790552</td>
      <td>3.790517</td>
      <td>3.790075</td>
      <td>3.790373</td>
      <td>3.790565</td>
      <td>3.790518</td>
      <td>3.790547</td>
      <td>3.790570</td>
      <td>3.790566</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>3.742230</td>
      <td>3.737859</td>
      <td>3.742241</td>
      <td>3.742058</td>
      <td>3.742187</td>
      <td>3.742228</td>
      <td>3.742240</td>
      <td>3.741911</td>
      <td>3.742093</td>
      <td>3.741819</td>
      <td>...</td>
      <td>3.741991</td>
      <td>3.742182</td>
      <td>3.742086</td>
      <td>3.739798</td>
      <td>3.741622</td>
      <td>3.742217</td>
      <td>3.742059</td>
      <td>3.742178</td>
      <td>3.742257</td>
      <td>3.742240</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>4.078794</td>
      <td>4.122720</td>
      <td>4.078413</td>
      <td>4.079142</td>
      <td>4.080528</td>
      <td>4.078799</td>
      <td>4.079100</td>
      <td>4.084911</td>
      <td>4.080099</td>
      <td>4.087041</td>
      <td>...</td>
      <td>4.082517</td>
      <td>4.079704</td>
      <td>4.083314</td>
      <td>4.125264</td>
      <td>4.082772</td>
      <td>4.077376</td>
      <td>4.082331</td>
      <td>4.079579</td>
      <td>4.078196</td>
      <td>4.078464</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>3.642450</td>
      <td>3.654639</td>
      <td>3.642340</td>
      <td>3.643128</td>
      <td>3.642614</td>
      <td>3.642305</td>
      <td>3.642277</td>
      <td>3.641956</td>
      <td>3.642936</td>
      <td>3.641301</td>
      <td>...</td>
      <td>3.642076</td>
      <td>3.642766</td>
      <td>3.642744</td>
      <td>3.665249</td>
      <td>3.643117</td>
      <td>3.642087</td>
      <td>3.643506</td>
      <td>3.642632</td>
      <td>3.642247</td>
      <td>3.642341</td>
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
    wzugmCevnXuCMCF4upAf0w    3.934835
    FhgAHo-8--equM8w5UZ41Q    3.788509
    IRIlwpomRvnXvpkeaGaM2A    3.763541
    46Ld9Qc9nAx_A0jwclNZiw    3.747641
    kKY726bQREexYHHNLK1H7g    3.746923
    _K63HbZBVQSBCvQicQdl-A    3.743300
    JN8s_dgw9nrSzkHnXxNOtg    3.741336
    9xSwne4GjwZ6Hlzdx2Zszg    3.728956
    y32M2Hkr7GsUqGG6KwOhZw    3.721766
    58APdML-PG_OD4El2ePTvw    3.716126
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
      <td>0.720741</td>
      <td>0.412029</td>
      <td>0.720665</td>
      <td>0.913932</td>
      <td>0.815495</td>
      <td>0.460906</td>
      <td>0.891103</td>
      <td>0.321446</td>
      <td>0.128758</td>
      <td>...</td>
      <td>0.375954</td>
      <td>0.595463</td>
      <td>0.697501</td>
      <td>0.289753</td>
      <td>0.515562</td>
      <td>0.015882</td>
      <td>0.810667</td>
      <td>0.730153</td>
      <td>0.474537</td>
      <td>0.494754</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>0.720741</td>
      <td>1.000000</td>
      <td>0.531490</td>
      <td>0.693314</td>
      <td>0.700075</td>
      <td>0.830051</td>
      <td>0.561576</td>
      <td>0.673440</td>
      <td>0.521506</td>
      <td>0.113499</td>
      <td>...</td>
      <td>0.338748</td>
      <td>0.963002</td>
      <td>0.818725</td>
      <td>0.829342</td>
      <td>0.670527</td>
      <td>-0.057316</td>
      <td>0.808700</td>
      <td>0.825599</td>
      <td>0.745091</td>
      <td>0.322130</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>0.412029</td>
      <td>0.531490</td>
      <td>1.000000</td>
      <td>0.916873</td>
      <td>0.631012</td>
      <td>0.715770</td>
      <td>0.959554</td>
      <td>0.682519</td>
      <td>0.970352</td>
      <td>0.104116</td>
      <td>...</td>
      <td>0.267434</td>
      <td>0.422807</td>
      <td>0.452622</td>
      <td>0.317162</td>
      <td>0.909885</td>
      <td>0.269327</td>
      <td>0.759309</td>
      <td>0.313519</td>
      <td>0.245549</td>
      <td>0.900771</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>0.720665</td>
      <td>0.693314</td>
      <td>0.916873</td>
      <td>1.000000</td>
      <td>0.855779</td>
      <td>0.911067</td>
      <td>0.896724</td>
      <td>0.882128</td>
      <td>0.849233</td>
      <td>0.120329</td>
      <td>...</td>
      <td>0.306988</td>
      <td>0.563970</td>
      <td>0.591316</td>
      <td>0.353743</td>
      <td>0.919337</td>
      <td>0.196887</td>
      <td>0.921728</td>
      <td>0.570053</td>
      <td>0.458413</td>
      <td>0.886281</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>0.913932</td>
      <td>0.700075</td>
      <td>0.631012</td>
      <td>0.855779</td>
      <td>1.000000</td>
      <td>0.826568</td>
      <td>0.600692</td>
      <td>0.873054</td>
      <td>0.599827</td>
      <td>-0.113172</td>
      <td>...</td>
      <td>0.187449</td>
      <td>0.549721</td>
      <td>0.541346</td>
      <td>0.209230</td>
      <td>0.710060</td>
      <td>-0.042394</td>
      <td>0.863534</td>
      <td>0.702391</td>
      <td>0.460178</td>
      <td>0.651209</td>
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
      <td>0.792956</td>
      <td>0.816564</td>
      <td>0.379237</td>
      <td>0.270042</td>
      <td>0.789788</td>
      <td>0.794940</td>
      <td>0.841365</td>
      <td>-0.042986</td>
      <td>0.743116</td>
      <td>...</td>
      <td>0.824476</td>
      <td>0.537650</td>
      <td>0.857398</td>
      <td>0.924114</td>
      <td>0.890950</td>
      <td>0.705694</td>
      <td>0.786955</td>
      <td>0.826806</td>
      <td>0.828485</td>
      <td>0.865850</td>
    </tr>
    <tr>
      <th>Dd-TkEszFMkSF-vRih51fQ</th>
      <td>0.792956</td>
      <td>1.000000</td>
      <td>0.993514</td>
      <td>0.366125</td>
      <td>0.308845</td>
      <td>0.983085</td>
      <td>0.981985</td>
      <td>0.885230</td>
      <td>-0.051364</td>
      <td>0.898010</td>
      <td>...</td>
      <td>0.995281</td>
      <td>0.790401</td>
      <td>0.907546</td>
      <td>0.954671</td>
      <td>0.811427</td>
      <td>0.314816</td>
      <td>0.701796</td>
      <td>0.991925</td>
      <td>0.656752</td>
      <td>0.877802</td>
    </tr>
    <tr>
      <th>YTdNcIWAt2nEzZ7NY-fniw</th>
      <td>0.816564</td>
      <td>0.993514</td>
      <td>1.000000</td>
      <td>0.373322</td>
      <td>0.264219</td>
      <td>0.989574</td>
      <td>0.977486</td>
      <td>0.913252</td>
      <td>-0.036936</td>
      <td>0.914095</td>
      <td>...</td>
      <td>0.990583</td>
      <td>0.813819</td>
      <td>0.913914</td>
      <td>0.961988</td>
      <td>0.826734</td>
      <td>0.369758</td>
      <td>0.746545</td>
      <td>0.994725</td>
      <td>0.713984</td>
      <td>0.904612</td>
    </tr>
    <tr>
      <th>bTRFge5pRWMh7IoCLn7lBw</th>
      <td>0.379237</td>
      <td>0.366125</td>
      <td>0.373322</td>
      <td>1.000000</td>
      <td>0.839735</td>
      <td>0.368340</td>
      <td>0.531877</td>
      <td>0.521869</td>
      <td>0.250402</td>
      <td>0.643674</td>
      <td>...</td>
      <td>0.382215</td>
      <td>0.505986</td>
      <td>0.478432</td>
      <td>0.429181</td>
      <td>0.520249</td>
      <td>0.541035</td>
      <td>0.645821</td>
      <td>0.389246</td>
      <td>0.582344</td>
      <td>0.429117</td>
    </tr>
    <tr>
      <th>-w7ww3yW5BHE3TFyj3IHuQ</th>
      <td>0.270042</td>
      <td>0.308845</td>
      <td>0.264219</td>
      <td>0.839735</td>
      <td>1.000000</td>
      <td>0.276764</td>
      <td>0.456102</td>
      <td>0.300552</td>
      <td>-0.040324</td>
      <td>0.475806</td>
      <td>...</td>
      <td>0.326871</td>
      <td>0.305280</td>
      <td>0.303664</td>
      <td>0.354267</td>
      <td>0.466218</td>
      <td>0.323128</td>
      <td>0.345939</td>
      <td>0.275958</td>
      <td>0.315474</td>
      <td>0.312387</td>
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
      <td>0.604929</td>
      <td>0.306082</td>
      <td>0.684301</td>
      <td>0.534033</td>
      <td>0.562720</td>
      <td>0.524335</td>
      <td>0.474005</td>
      <td>0.543050</td>
      <td>0.687466</td>
      <td>...</td>
      <td>0.869826</td>
      <td>0.768017</td>
      <td>-0.103955</td>
      <td>0.787516</td>
      <td>0.251441</td>
      <td>0.146670</td>
      <td>0.373930</td>
      <td>0.649073</td>
      <td>0.811284</td>
      <td>0.559290</td>
    </tr>
    <tr>
      <th>8Rdz0VPY8CuT2GQZ7ho2sw</th>
      <td>0.604929</td>
      <td>1.000000</td>
      <td>0.771181</td>
      <td>0.668666</td>
      <td>0.446506</td>
      <td>0.806925</td>
      <td>0.407330</td>
      <td>0.237145</td>
      <td>0.190323</td>
      <td>0.863641</td>
      <td>...</td>
      <td>0.625452</td>
      <td>0.585623</td>
      <td>0.172381</td>
      <td>0.678374</td>
      <td>0.591647</td>
      <td>0.155667</td>
      <td>0.813378</td>
      <td>0.515024</td>
      <td>0.721343</td>
      <td>0.824730</td>
    </tr>
    <tr>
      <th>DAMTCTsSeACXbkSABkhZqQ</th>
      <td>0.306082</td>
      <td>0.771181</td>
      <td>1.000000</td>
      <td>0.602944</td>
      <td>0.192602</td>
      <td>0.909381</td>
      <td>0.311445</td>
      <td>0.273086</td>
      <td>-0.225258</td>
      <td>0.532435</td>
      <td>...</td>
      <td>0.507281</td>
      <td>0.601425</td>
      <td>0.140032</td>
      <td>0.577847</td>
      <td>0.582765</td>
      <td>0.133754</td>
      <td>0.715401</td>
      <td>0.037360</td>
      <td>0.430301</td>
      <td>0.808834</td>
    </tr>
    <tr>
      <th>6I6uDGwCDggrWXi2T4lfaA</th>
      <td>0.684301</td>
      <td>0.668666</td>
      <td>0.602944</td>
      <td>1.000000</td>
      <td>0.601799</td>
      <td>0.764556</td>
      <td>0.579640</td>
      <td>0.466201</td>
      <td>0.330502</td>
      <td>0.823990</td>
      <td>...</td>
      <td>0.905221</td>
      <td>0.948628</td>
      <td>0.225421</td>
      <td>0.931421</td>
      <td>0.803133</td>
      <td>0.177240</td>
      <td>0.568698</td>
      <td>0.372070</td>
      <td>0.904395</td>
      <td>0.768868</td>
    </tr>
    <tr>
      <th>qUdGBSFkiPhEL6I718y-Gg</th>
      <td>0.534033</td>
      <td>0.446506</td>
      <td>0.192602</td>
      <td>0.601799</td>
      <td>1.000000</td>
      <td>0.215027</td>
      <td>0.283366</td>
      <td>0.191661</td>
      <td>0.680100</td>
      <td>0.482624</td>
      <td>...</td>
      <td>0.702884</td>
      <td>0.575249</td>
      <td>0.144105</td>
      <td>0.644695</td>
      <td>0.478036</td>
      <td>-0.127833</td>
      <td>0.296017</td>
      <td>0.236424</td>
      <td>0.727604</td>
      <td>0.681321</td>
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
      <td>0.178899</td>
      <td>0.951031</td>
      <td>0.691397</td>
      <td>0.259757</td>
      <td>0.940552</td>
      <td>0.816696</td>
      <td>0.776824</td>
      <td>0.535893</td>
      <td>0.260423</td>
      <td>...</td>
      <td>-0.007849</td>
      <td>0.824773</td>
      <td>0.390546</td>
      <td>0.062772</td>
      <td>0.329732</td>
      <td>0.578336</td>
      <td>0.915200</td>
      <td>0.217883</td>
      <td>0.971168</td>
      <td>0.982351</td>
    </tr>
    <tr>
      <th>DAMTCTsSeACXbkSABkhZqQ</th>
      <td>0.178899</td>
      <td>1.000000</td>
      <td>0.210156</td>
      <td>0.657375</td>
      <td>0.310377</td>
      <td>0.164341</td>
      <td>0.301864</td>
      <td>0.461970</td>
      <td>0.494442</td>
      <td>0.645157</td>
      <td>...</td>
      <td>0.623212</td>
      <td>0.415439</td>
      <td>0.526627</td>
      <td>0.370224</td>
      <td>0.874367</td>
      <td>0.416135</td>
      <td>0.267851</td>
      <td>0.607619</td>
      <td>0.164944</td>
      <td>0.169984</td>
    </tr>
    <tr>
      <th>6I6uDGwCDggrWXi2T4lfaA</th>
      <td>0.951031</td>
      <td>0.210156</td>
      <td>1.000000</td>
      <td>0.749799</td>
      <td>0.151422</td>
      <td>0.900130</td>
      <td>0.774837</td>
      <td>0.741639</td>
      <td>0.504730</td>
      <td>0.219069</td>
      <td>...</td>
      <td>-0.052070</td>
      <td>0.696736</td>
      <td>0.268463</td>
      <td>-0.109877</td>
      <td>0.366669</td>
      <td>0.718121</td>
      <td>0.814460</td>
      <td>0.064799</td>
      <td>0.984445</td>
      <td>0.990368</td>
    </tr>
    <tr>
      <th>qUdGBSFkiPhEL6I718y-Gg</th>
      <td>0.691397</td>
      <td>0.657375</td>
      <td>0.749799</td>
      <td>1.000000</td>
      <td>0.193305</td>
      <td>0.649815</td>
      <td>0.592183</td>
      <td>0.545754</td>
      <td>0.660185</td>
      <td>0.520032</td>
      <td>...</td>
      <td>0.320945</td>
      <td>0.654766</td>
      <td>0.337880</td>
      <td>0.193709</td>
      <td>0.740265</td>
      <td>0.487023</td>
      <td>0.675719</td>
      <td>0.436975</td>
      <td>0.703693</td>
      <td>0.708155</td>
    </tr>
    <tr>
      <th>ujcbqs6jZfaESgSLvbjWuQ</th>
      <td>0.259757</td>
      <td>0.310377</td>
      <td>0.151422</td>
      <td>0.193305</td>
      <td>1.000000</td>
      <td>0.164879</td>
      <td>0.139443</td>
      <td>0.592555</td>
      <td>0.200928</td>
      <td>0.207123</td>
      <td>...</td>
      <td>-0.063087</td>
      <td>0.316322</td>
      <td>0.884324</td>
      <td>0.251879</td>
      <td>0.063906</td>
      <td>0.038891</td>
      <td>0.210878</td>
      <td>0.226732</td>
      <td>0.215547</td>
      <td>0.201362</td>
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


    CF Train RMSE score is 0.09213448455145726
    CF Test RMSE score is 0.05653953923441608

