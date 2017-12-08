---
nav_include: 2
title: Filter-data-by-city
notebook: filtering-by-city.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}



```python
#Set up workspace
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
import os
import json
```




```python
#Reading in dataframes:
users = pd.read_json("user.json",lines=True) #users
businesses = pd.read_json("business.json",lines=True) #businesses

with open('review.json') as json_file:      #reviews
    reviews = json_file.readlines()
    reviews = list(map(json.loads, reviews)) 
pd.DataFrame(reviews)
reviews_df=pd.DataFrame(reviews,columns=['business_id','cool','date','funny','review_id','stars','text','useful','user_id'])
```




```python
#Formatting users dataset
users['elite'] = [x if len(x)>0 else 'NaN' for x in users['elite']]

elite_status = [] #Creating a new elite status column
for i in users['elite']:
    if i=='NaN':
        elite_status.append('No')
    else:
        elite_status.append('Yes')
users = users.assign(elite_status = elite_status)

#Formatting datetime
users['yelping_since'] = [datetime.strptime(i, '%Y-%m-%d') for i in users['yelping_since']]
```




```python
#Filtering businesses dataset for just restaurants
new_column = []
for j in businesses['categories']:
    if j.count('Restaurants')>0:
        new_column.append('Yes')
    else:
        new_column.append('No')
businesses['Restaurant_Status']=new_column

businesses = businesses[businesses['Restaurant_Status']=='Yes']
```




```python
#Formula to define dataframes filtered by city
def choose_city(city,review_count=5):      
    city_businesses = businesses[businesses['city']==city]
    city_businesses = city_businesses[city_businesses['review_count']>(review_count - 1)]

    #Subsetting reviews dataset
    business_list = list(set(city_businesses['business_id']))
    city_reviews = reviews_df[reviews_df['business_id'].isin(business_list)]
    keep_bids = []
    for bid, df in city_reviews.groupby('business_id'):
        if df.shape[0]>=review_count:
            keep_bids.append(bid)            
    city_reviews = city_reviews[city_reviews['business_id'].isin(keep_bids)]

    #users
    city_user_list = list(set(city_reviews['user_id']))
    city_users = users[users['user_id'].isin(city_user_list)] 
    city_user_list = list(set(city_users['user_id']))    
    city_reviews = city_reviews[city_reviews['user_id'].isin(city_user_list)]    
    
    #Saving each file to .csv
    city_reviews.to_json(str(city)+'_reviews.json')
    city_users.to_json(str(city)+'_users.json')
    city_businesses.to_json(str(city)+'_businesses.json')
        
    return {'businesses':city_businesses, 'users':city_users ,'reviews':city_reviews }
```


Next, create an ordered list of cities by number of reviews. From this ordered list we'll select cities with a moderate but not excessive number of reviews to be our sample set. Then we'll save the dataframes for each of these cities to .csv's so that we can just read these in in future, without having to do the entire set of steps above.  



```python
#count cities
cities = list(set(businesses['city']))
cities = list(filter(None, cities))

business_counts = []
for i in cities:
    businesses_list = list(businesses['city'])
    business_counts.append(businesses_list.count(i))

cities_df = pd.DataFrame({
        'Cities': cities,
        'Businesses': business_counts})

cities_df = cities_df.sort_values(by=['Businesses'],ascending=False)

medium_cities = list(cities_df['Cities'][25:50])
```




```python
medium_cities
```




```python

#looping through to create many different cities dfs:
cities_dfs = []
for i in ['Montreal']:
    cities_dfs.append(choose_city(i))
dfs_dict = dict(zip(medium_cities,cities_dfs))
```

