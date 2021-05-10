# Imports 

## Packages


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#the inline magic line is needed to make some data visualizations come out right when you are using jupyter notebooks
%matplotlib inline 
import seaborn as sns

from math import sqrt
```


```python
#ML algorithms (estimators)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model

#model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

#cross validation
from sklearn.model_selection import train_test_split

"""sklearn imports from classification problem in course 1"""
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report
```




    'sklearn imports from classification problem in course 1'



## Import & Verify Data


```python
#import numerical only data
df= pd.read_csv('num_only_credit_one_data.csv')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>SEX</th>
      <th>AGE</th>
      <th>LIMIT_BAL</th>
      <th>bill_ap</th>
      <th>bill_m</th>
      <th>bill_ju</th>
      <th>bill_jy</th>
      <th>bill_ag</th>
      <th>bill_s</th>
      <th>...</th>
      <th>pay_s</th>
      <th>DEFAULT</th>
      <th>edu_graduate school</th>
      <th>edu_high school</th>
      <th>edu_other</th>
      <th>edu_university</th>
      <th>mar_0</th>
      <th>mar_1</th>
      <th>mar_2</th>
      <th>mar_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>20000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>3102</td>
      <td>3913</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>26</td>
      <td>120000</td>
      <td>3261</td>
      <td>3455</td>
      <td>3272</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>...</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>34</td>
      <td>90000</td>
      <td>15549</td>
      <td>14948</td>
      <td>14331</td>
      <td>13559</td>
      <td>14027</td>
      <td>29239</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>37</td>
      <td>50000</td>
      <td>29547</td>
      <td>28959</td>
      <td>28314</td>
      <td>49291</td>
      <td>48233</td>
      <td>46990</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>57</td>
      <td>50000</td>
      <td>19131</td>
      <td>19146</td>
      <td>20940</td>
      <td>35835</td>
      <td>5670</td>
      <td>8617</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
df.shape
```




    (30000, 31)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 31 columns):
     #   Column               Non-Null Count  Dtype
    ---  ------               --------------  -----
     0   ID                   30000 non-null  int64
     1   SEX                  30000 non-null  int64
     2   AGE                  30000 non-null  int64
     3   LIMIT_BAL            30000 non-null  int64
     4   bill_ap              30000 non-null  int64
     5   bill_m               30000 non-null  int64
     6   bill_ju              30000 non-null  int64
     7   bill_jy              30000 non-null  int64
     8   bill_ag              30000 non-null  int64
     9   bill_s               30000 non-null  int64
     10  pmt_ap               30000 non-null  int64
     11  pmt_m                30000 non-null  int64
     12  pmt_ju               30000 non-null  int64
     13  pmt_jy               30000 non-null  int64
     14  pmt_ag               30000 non-null  int64
     15  pmt_s                30000 non-null  int64
     16  pay_ap               30000 non-null  int64
     17  pay_m                30000 non-null  int64
     18  pay_ju               30000 non-null  int64
     19  pay_jy               30000 non-null  int64
     20  pay_ag               30000 non-null  int64
     21  pay_s                30000 non-null  int64
     22  DEFAULT              30000 non-null  int64
     23  edu_graduate school  30000 non-null  int64
     24  edu_high school      30000 non-null  int64
     25  edu_other            30000 non-null  int64
     26  edu_university       30000 non-null  int64
     27  mar_0                30000 non-null  int64
     28  mar_1                30000 non-null  int64
     29  mar_2                30000 non-null  int64
     30  mar_3                30000 non-null  int64
    dtypes: int64(31)
    memory usage: 7.1 MB
    

Everything with my dataset looks as I expect. So I will move on to specifying the variables I was to use as features (IVs) and the one I want to use as target for my model predicting credit limit. 

# Predicting Customer Credit Limits
In this notebook I am only going to create and test models that attempt to predict customer credit limits. Which means the target variable in all models will be 'limit' (i.e. limit_bal).

#### Re-ordering df Columns
To make things easier, before doing anything I'm going to move LIMIT_BAL to the last column in my df and rename it. 


```python
#step 1: create a variabled to hold the LIMIT_BAL var
lim=df['LIMIT_BAL']

#step 2: drop the LIMIT_BAL column
df.drop(labels='LIMIT_BAL', axis=1, inplace=True)

#step 3: insert 'lim' as last column of df
df.insert(loc=30, column='limit', value=lim)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>SEX</th>
      <th>AGE</th>
      <th>bill_ap</th>
      <th>bill_m</th>
      <th>bill_ju</th>
      <th>bill_jy</th>
      <th>bill_ag</th>
      <th>bill_s</th>
      <th>pmt_ap</th>
      <th>...</th>
      <th>DEFAULT</th>
      <th>edu_graduate school</th>
      <th>edu_high school</th>
      <th>edu_other</th>
      <th>edu_university</th>
      <th>mar_0</th>
      <th>mar_1</th>
      <th>mar_2</th>
      <th>mar_3</th>
      <th>limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>3102</td>
      <td>3913</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>20000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>26</td>
      <td>3261</td>
      <td>3455</td>
      <td>3272</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>2000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>34</td>
      <td>15549</td>
      <td>14948</td>
      <td>14331</td>
      <td>13559</td>
      <td>14027</td>
      <td>29239</td>
      <td>5000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>90000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>37</td>
      <td>29547</td>
      <td>28959</td>
      <td>28314</td>
      <td>49291</td>
      <td>48233</td>
      <td>46990</td>
      <td>1000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>57</td>
      <td>19131</td>
      <td>19146</td>
      <td>20940</td>
      <td>35835</td>
      <td>5670</td>
      <td>8617</td>
      <td>679</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>50000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### Specify the Target Variable
Now I will specify the dependent or target variable for models attempting to predict customer credit limits. 


```python
#filter the data to return my dependent var (the limit column)
y = df.loc[:,'limit']

#verify that the first 5 values in y match the values in the limit column above
y.head()
```




    0     20000
    1    120000
    2     90000
    3     50000
    4     50000
    Name: limit, dtype: int64




```python
#verify that y has 30,000 rows
y.shape
```




    (30000,)



# Model 1: Predicting Credit Limit Using All Variables
None of the variables stood out as ones that should be included or excluded from the model, so in this first model I'm going to include them all.

### Specify the Feature Variables


```python
#filter the data to return only columns for my feature vars
X = df.iloc[:, 1:30]

#verify that all but the ID and limit columns were selected
print('Summary of Feature Variables') 
X.head()
```

    Summary of Feature Variables
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEX</th>
      <th>AGE</th>
      <th>bill_ap</th>
      <th>bill_m</th>
      <th>bill_ju</th>
      <th>bill_jy</th>
      <th>bill_ag</th>
      <th>bill_s</th>
      <th>pmt_ap</th>
      <th>pmt_m</th>
      <th>...</th>
      <th>pay_s</th>
      <th>DEFAULT</th>
      <th>edu_graduate school</th>
      <th>edu_high school</th>
      <th>edu_other</th>
      <th>edu_university</th>
      <th>mar_0</th>
      <th>mar_1</th>
      <th>mar_2</th>
      <th>mar_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>3102</td>
      <td>3913</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>26</td>
      <td>3261</td>
      <td>3455</td>
      <td>3272</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>2000</td>
      <td>0</td>
      <td>...</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>34</td>
      <td>15549</td>
      <td>14948</td>
      <td>14331</td>
      <td>13559</td>
      <td>14027</td>
      <td>29239</td>
      <td>5000</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>37</td>
      <td>29547</td>
      <td>28959</td>
      <td>28314</td>
      <td>49291</td>
      <td>48233</td>
      <td>46990</td>
      <td>1000</td>
      <td>1069</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>57</td>
      <td>19131</td>
      <td>19146</td>
      <td>20940</td>
      <td>35835</td>
      <td>5670</td>
      <td>8617</td>
      <td>679</td>
      <td>689</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



## Train & Test Model 1 Using 3 Different Regression Algorithms
Because credit limit is a continuous variable, I know I will have to use a regression model to predict it. But I don't have a principled reason to choose one regression algorithm over others, so I'll compare the predictions of models using Random Forest p9idifferent regression algorithms to find the one that works best on this task. 
* I will run each model through a 3-fold cross validation test. Using R squared to score the models. 
* The model with the highest mean CV sore will be labeled best.

### Setup


```python
#create a dictionary to hold the algos
algosClass = [ ]

#add the randomn forest regressor to the dictionary
algosClass.append(('Random Forest Regressor',RandomForestRegressor()))

#add the liner regression algo to the dictionary
algosClass.append(('Linear Regression',LinearRegression()))

#add the SV regressor to the dictionary
algosClass.append(('Support Vector Regression',SVR()))

#verify every algo was added to the list
print(algosClass)
```

    [('Random Forest Regressor', RandomForestRegressor()), ('Linear Regression', LinearRegression()), ('Support Vector Regression', SVR())]
    

### Train & Test Models


```python
#create two empty lists to hold the names and cross validation results from each model
results = []
names = []

#create a for-loop that will
for name, model in algosClass:
    result = cross_val_score(model, X,y, cv=3, scoring='r2')
    names.append(name)
    results.append(result)
```

### Performance Results


```python
#run the for loop and print the mean cross validation score for each model
for i in range(len(names)):
    print(names[i],results[i].mean())
```

    Random Forest Regressor 0.46700202592747336
    Linear Regression 0.35871695390197167
    Support Vector Regression -0.05039128320492573
    

The model built using the random forest regressor performed somewhat less badly than the other two models. However, even the randomn forest model does little better than chance at predicting customer's credit limits (i.e.limit_bal). 

# Credit Limit Model 2: Excluding the Demographic Variables 

## Select the Feature Variables


```python
#filter the data to return only columns for bill amount, payment amount, payment history, and default var
X2 = df.iloc[:, 3:22]

print('Summary of Model 2 Features')
X2.head()
```

    Summary of Features for Model 2
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bill_ap</th>
      <th>bill_m</th>
      <th>bill_ju</th>
      <th>bill_jy</th>
      <th>bill_ag</th>
      <th>bill_s</th>
      <th>pmt_ap</th>
      <th>pmt_m</th>
      <th>pmt_ju</th>
      <th>pmt_jy</th>
      <th>pmt_ag</th>
      <th>pmt_s</th>
      <th>pay_ap</th>
      <th>pay_m</th>
      <th>pay_ju</th>
      <th>pay_jy</th>
      <th>pay_ag</th>
      <th>pay_s</th>
      <th>DEFAULT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>3102</td>
      <td>3913</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>-2</td>
      <td>-2</td>
      <td>-1</td>
      <td>-1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3261</td>
      <td>3455</td>
      <td>3272</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>2000</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15549</td>
      <td>14948</td>
      <td>14331</td>
      <td>13559</td>
      <td>14027</td>
      <td>29239</td>
      <td>5000</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>1500</td>
      <td>1518</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29547</td>
      <td>28959</td>
      <td>28314</td>
      <td>49291</td>
      <td>48233</td>
      <td>46990</td>
      <td>1000</td>
      <td>1069</td>
      <td>1100</td>
      <td>1200</td>
      <td>2019</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19131</td>
      <td>19146</td>
      <td>20940</td>
      <td>35835</td>
      <td>5670</td>
      <td>8617</td>
      <td>679</td>
      <td>689</td>
      <td>9000</td>
      <td>10000</td>
      <td>36681</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create two empty lists to hold the names and cross validation results from each model
results2 = []
names2 = []

#get cross validation scores for model 2 
for name, model in algosClass:
    result = cross_val_score(model, X2,y, cv=3, scoring='r2')
    names2.append(name)
    results2.append(result)
```


```python
#run the for loop and print the mean cross validation score for each model
print("Summary of Model2 Cross Validation Scores")

for i in range(len(names2)):
    print(names[i],results[i].mean())
```

    Summary of Model2 Cross Validation Scores
    Random Forest Regressor 0.46700202592747336
    Linear Regression 0.35871695390197167
    Support Vector Regression -0.05039128320492573
    

## Train & Test the Model


```python
#instantiate the random forest regression algo
# rfr = RandomForestRegressor()
```


```python
#train the rfr model on X_train and y_train
# model = rfr.fit(X_train, y_train)

#use trained model to predict credit limit amounts for the X_test data
# y_predict = model_1.predict(X_test)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-15-966390a2adc9> in <module>
          1 #train the rfr model on X_train and y_train
    ----> 2 model = rfr.fit(X_train, y_train)
          3 
          4 #use trained model to predict credit limit amounts for the X_test data
          5 y_predict = model_1.predict(X_test)
    

    NameError: name 'X_train' is not defined


## Split Data into Training and Testing Sets


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)
```

>*Coding Notes*
>* The ```test_size =``` parameter specifies what percent of you dataset will be used for training and training. If you don't specify an amount, the default is 0.25.
>* The ```random_state =``` parameter can be set to any number. The purpose of it is to make sure that every time the model is run,  the same observations are used in the training and testing sets. If this parameter is not included, every time you run the model it may select a slightly different set of observations.


```python
cross_val_score(rfr, X, y, cv=3)
```


```python
r2_score(y_test, y_predict)
```


```python
mean_squared_error(y_test, y_predict)
```


```python

```
