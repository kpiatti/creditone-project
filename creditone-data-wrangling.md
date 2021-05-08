### Imports and Settings


```python
from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# Obtain Raw Data  
The Credit One customer data is on the MySQL sever. So in this stage I have to connect to the server, query the database for the customer data, and set it up as a Pandas dataframe, and then export the raw data as a .csv file. 

## Connect to MySQL & Query Database


```python
#connect python to MySQL server (database?)
db_connection_str = 'mysql+pymysql://deepanalytics:Sqltask1234!@34.73.222.197/deepanalytics'
```


```python
#create engine for SQLalchemy to interface with the database API (see notes below)
db_connection = create_engine(db_connection_str)

# first time successfully queried db without this line, so not entirely
#sure why it is needed. asked on slack, answer = link to oriely article

```


```python
#query the credit one data and extract it into a pd df
sql_query = pd.read_sql('SELECT * FROM credit', con=db_connection)
```


```python
#create pandas df from SQL query
df = pd.DataFrame(sql_query)
```

*Notes*
* First time through the exercise, I missed the ```db_connection = create_engine(db_connection_str)``` line. i got a name error b/c 'db_connection' was not defined. so i removed the _str from the first line, ran the cells, and was able to execute the query in my notebook. So ultimately, I'm unsure why we need the create_engine code. (see my SQL collection in edge for some explanatory links)
* in the course outline, they did not have us use ```pd.DataFrame()```. They just told us to create the df from ```df = pd.read_sql()```. I suspect that was an oversight.
* I made the slight changes you see following a tutorial I found on how to export a SQL query as a csv file.

## Verify Proper Extraction


```python
df.head().to_markdown
```




    <bound method DataFrame.to_markdown of   MyUnknownColumn         X1      X2          X3        X4   X5     X6     X7  \
    0              ID  LIMIT_BAL     SEX   EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2   
    1               1      20000  female  university         1   24      2      2   
    2               2     120000  female  university         2   26     -1      2   
    3               3      90000  female  university         2   34      0      0   
    4               4      50000  female  university         1   37      0      0   
    
          X8     X9  ...        X15        X16        X17       X18       X19  \
    0  PAY_3  PAY_4  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2   
    1     -1     -1  ...          0          0          0         0       689   
    2      0      0  ...       3272       3455       3261         0      1000   
    3      0      0  ...      14331      14948      15549      1518      1500   
    4      0      0  ...      28314      28959      29547      2000      2019   
    
            X20       X21       X22       X23                           Y  
    0  PAY_AMT3  PAY_AMT4  PAY_AMT5  PAY_AMT6  default payment next month  
    1         0         0         0         0                     default  
    2      1000      1000         0      2000                     default  
    3      1000      1000      1000      5000                 not default  
    4      1200      1100      1069      1000                 not default  
    
    [5 rows x 25 columns]>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30204 entries, 0 to 30203
    Data columns (total 25 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   MyUnknownColumn  30204 non-null  object
     1   X1               30204 non-null  object
     2   X2               30204 non-null  object
     3   X3               30204 non-null  object
     4   X4               30204 non-null  object
     5   X5               30204 non-null  object
     6   X6               30204 non-null  object
     7   X7               30204 non-null  object
     8   X8               30204 non-null  object
     9   X9               30204 non-null  object
     10  X10              30204 non-null  object
     11  X11              30204 non-null  object
     12  X12              30204 non-null  object
     13  X13              30204 non-null  object
     14  X14              30204 non-null  object
     15  X15              30204 non-null  object
     16  X16              30204 non-null  object
     17  X17              30204 non-null  object
     18  X18              30204 non-null  object
     19  X19              30204 non-null  object
     20  X20              30204 non-null  object
     21  X21              30204 non-null  object
     22  X22              30204 non-null  object
     23  X23              30204 non-null  object
     24  Y                30204 non-null  object
    dtypes: object(25)
    memory usage: 5.8+ MB
    

## Export Raw Data As CSV File


```python
#export df as .csv file
df.to_csv (r'C:\Users\Kpiat\export_data.csv', index = False)
```

*Notes*  

* This data is from a paper published in Expert Systems with Applications found [here](https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf)
* A short summary of project and description of variables is in PDF in course 2 folder.
* Dataframe Characteristics: 26 columns—23 features/independent variables, 1 dependent variable, column 0 = ID, and 30,204 rows 
* Initial Cleaning Tasks:
1. Reset column headings/names to strings in row 0 
2. Delete row 0 from the dataframe.
3. Change dtypes for variables for object to appropriate type
4. First 200+ rows may be duplicates, investigate furthur. 

---

# Process Data


```python
#import exported csv file with raw data
df = pd.read_csv('raw_credit_one_data.csv')
```


```python
#verify imported data looks as expected 
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
      <th>MyUnknownColumn</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>...</th>
      <th>X15</th>
      <th>X16</th>
      <th>X17</th>
      <th>X18</th>
      <th>X19</th>
      <th>X20</th>
      <th>X21</th>
      <th>X22</th>
      <th>X23</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID</td>
      <td>LIMIT_BAL</td>
      <td>SEX</td>
      <td>EDUCATION</td>
      <td>MARRIAGE</td>
      <td>AGE</td>
      <td>PAY_0</td>
      <td>PAY_2</td>
      <td>PAY_3</td>
      <td>PAY_4</td>
      <td>...</td>
      <td>BILL_AMT4</td>
      <td>BILL_AMT5</td>
      <td>BILL_AMT6</td>
      <td>PAY_AMT1</td>
      <td>PAY_AMT2</td>
      <td>PAY_AMT3</td>
      <td>PAY_AMT4</td>
      <td>PAY_AMT5</td>
      <td>PAY_AMT6</td>
      <td>default payment next month</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>120000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>90000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>50000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
#look at number of rows and columns in df as benchmark before making any changes 
df.shape
```




    (30204, 25)



## Change Column Headings 


```python
#change column headings to values in index (row) 0 
df.columns = df.iloc[0]
```


```python
#verify change was successful
df.head().to_markdown
```




    <bound method DataFrame.to_markdown of 0  ID  LIMIT_BAL     SEX   EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  \
    0  ID  LIMIT_BAL     SEX   EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3   
    1   1      20000  female  university         1   24      2      2     -1   
    2   2     120000  female  university         2   26     -1      2      0   
    3   3      90000  female  university         2   34      0      0      0   
    4   4      50000  female  university         1   37      0      0      0   
    
    0  PAY_4  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  \
    0  PAY_4  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3   
    1     -1  ...          0          0          0         0       689         0   
    2      0  ...       3272       3455       3261         0      1000      1000   
    3      0  ...      14331      14948      15549      1518      1500      1000   
    4      0  ...      28314      28959      29547      2000      2019      1200   
    
    0  PAY_AMT4  PAY_AMT5  PAY_AMT6  default payment next month  
    0  PAY_AMT4  PAY_AMT5  PAY_AMT6  default payment next month  
    1         0         0         0                     default  
    2      1000         0      2000                     default  
    3      1000      1000      5000                 not default  
    4      1100      1069      1000                 not default  
    
    [5 rows x 25 columns]>




```python
#drop row 0 that contains column headings
df.drop(df.index[0], inplace = True)
```


```python
#verify df is -1 row
len(df)
```




    30203




```python
#verify column headings row was dropped
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
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>120000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>90000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>50000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>50000</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>not default</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



*Notes*
* Found how to do this on [Stack Overflow](https://stackoverflow.com/questions/26147180/convert-row-to-column-header-for-pandas-dataframe).
* ```df.iloc[pd.RangeIndex(len(df)).drop(0)]``` is for situations where your df does not have or you don't know if it has unique index numbers for each row. If you know all index numbers are unique, you can use ```df.drop(df.index[0])```.
* I choose non-unique method because I'm unsure what the situation is. 

***♠ Why is the 0 index missing? I thought df.drop would drop data from index 0 and replace it with values that used to be in index 1.***


```python
#examine rows 200-206 of df
df.loc[200:206]
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
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>200</th>
      <td>200</td>
      <td>30000</td>
      <td>female</td>
      <td>high school</td>
      <td>2</td>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>29836</td>
      <td>1630</td>
      <td>0</td>
      <td>1000</td>
      <td>85</td>
      <td>1714</td>
      <td>104</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>201</th>
      <td>201</td>
      <td>180000</td>
      <td>female</td>
      <td>graduate school</td>
      <td>1</td>
      <td>38</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
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
      <td>not default</td>
    </tr>
    <tr>
      <th>202</th>
      <td>NaN</td>
      <td>X1</td>
      <td>X2</td>
      <td>X3</td>
      <td>X4</td>
      <td>X5</td>
      <td>X6</td>
      <td>X7</td>
      <td>X8</td>
      <td>X9</td>
      <td>...</td>
      <td>X15</td>
      <td>X16</td>
      <td>X17</td>
      <td>X18</td>
      <td>X19</td>
      <td>X20</td>
      <td>X21</td>
      <td>X22</td>
      <td>X23</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>203</th>
      <td>ID</td>
      <td>LIMIT_BAL</td>
      <td>SEX</td>
      <td>EDUCATION</td>
      <td>MARRIAGE</td>
      <td>AGE</td>
      <td>PAY_0</td>
      <td>PAY_2</td>
      <td>PAY_3</td>
      <td>PAY_4</td>
      <td>...</td>
      <td>BILL_AMT4</td>
      <td>BILL_AMT5</td>
      <td>BILL_AMT6</td>
      <td>PAY_AMT1</td>
      <td>PAY_AMT2</td>
      <td>PAY_AMT3</td>
      <td>PAY_AMT4</td>
      <td>PAY_AMT5</td>
      <td>PAY_AMT6</td>
      <td>default payment next month</td>
    </tr>
    <tr>
      <th>204</th>
      <td>1</td>
      <td>20000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>205</th>
      <td>2</td>
      <td>120000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>206</th>
      <td>3</td>
      <td>90000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 25 columns</p>
</div>



## Wrangle Duplicate Data

***♠ How do we look for duplicate columns? Do we need to do that? How would we drop duplicate columns?***

### Duplicate Rows


```python
#find and print all duplicate rows in the df
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)
```

    0     ID LIMIT_BAL     SEX        EDUCATION MARRIAGE AGE PAY_0 PAY_2 PAY_3  \
    204    1     20000  female       university        1  24     2     2    -1   
    205    2    120000  female       university        2  26    -1     2     0   
    206    3     90000  female       university        2  34     0     0     0   
    207    4     50000  female       university        1  37     0     0     0   
    208    5     50000    male       university        1  57    -1     0    -1   
    ..   ...       ...     ...              ...      ...  ..   ...   ...   ...   
    400  197    150000  female       university        1  34    -2    -2    -2   
    401  198     20000  female  graduate school        2  22     0     0     0   
    402  199    500000  female  graduate school        1  34    -2    -2    -2   
    403  200     30000  female      high school        2  22     1     2     2   
    404  201    180000  female  graduate school        1  38    -2    -2    -2   
    
    0   PAY_4  ... BILL_AMT4 BILL_AMT5 BILL_AMT6 PAY_AMT1 PAY_AMT2 PAY_AMT3  \
    204    -1  ...         0         0         0        0      689        0   
    205     0  ...      3272      3455      3261        0     1000     1000   
    206     0  ...     14331     14948     15549     1518     1500     1000   
    207     0  ...     28314     28959     29547     2000     2019     1200   
    208     0  ...     20940     19146     19131     2000    36681    10000   
    ..    ...  ...       ...       ...       ...      ...      ...      ...   
    400    -2  ...       116         0      1500        0        0      116   
    401     0  ...      8332     18868     19247     1500     1032      541   
    402    -1  ...      1251      1206      1151      138     2299     1251   
    403     0  ...     29836      1630         0     1000       85     1714   
    404    -2  ...         0         0         0        0        0        0   
    
    0   PAY_AMT4 PAY_AMT5 PAY_AMT6 default payment next month  
    204        0        0        0                    default  
    205     1000        0     2000                    default  
    206     1000     1000     5000                not default  
    207     1100     1069     1000                not default  
    208     9000      689      679                not default  
    ..       ...      ...      ...                        ...  
    400        0     1500        0                not default  
    401    20000      693     1000                not default  
    402     1206     1151    15816                not default  
    403      104        0        0                    default  
    404        0        0        0                not default  
    
    [201 rows x 25 columns]
    


```python
#drop duplicate rows from the df
df.drop_duplicates(inplace = True)  #inplace=True is to make change to df permanent
```


```python
#verify rows were dropped
len(df)
```




    30002



## Missing Data


```python
#check for missing values
print(df.isnull().sum())
```

    0
    ID                            1
    LIMIT_BAL                     0
    SEX                           0
    EDUCATION                     0
    MARRIAGE                      0
    AGE                           0
    PAY_0                         0
    PAY_2                         0
    PAY_3                         0
    PAY_4                         0
    PAY_5                         0
    PAY_6                         0
    BILL_AMT1                     0
    BILL_AMT2                     0
    BILL_AMT3                     0
    BILL_AMT4                     0
    BILL_AMT5                     0
    BILL_AMT6                     0
    PAY_AMT1                      0
    PAY_AMT2                      0
    PAY_AMT3                      0
    PAY_AMT4                      0
    PAY_AMT5                      0
    PAY_AMT6                      0
    default payment next month    0
    dtype: int64
    

*Notes*
* There appears to be one one missing data point in the ID var.
* However, I'm not sure running ```df.isnull()``` is enough to check for that.


```python
#drop the row with missing ID
df.dropna(inplace=True)
```


```python
len(df)
```




    30001



---

## Invaild Data


```python
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
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>120000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>90000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>50000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>50000</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>not default</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
#locate rows 201-206 (not inclusive)
df.loc[201:206]
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
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>201</th>
      <td>201</td>
      <td>180000</td>
      <td>female</td>
      <td>graduate school</td>
      <td>1</td>
      <td>38</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
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
      <td>not default</td>
    </tr>
    <tr>
      <th>203</th>
      <td>ID</td>
      <td>LIMIT_BAL</td>
      <td>SEX</td>
      <td>EDUCATION</td>
      <td>MARRIAGE</td>
      <td>AGE</td>
      <td>PAY_0</td>
      <td>PAY_2</td>
      <td>PAY_3</td>
      <td>PAY_4</td>
      <td>...</td>
      <td>BILL_AMT4</td>
      <td>BILL_AMT5</td>
      <td>BILL_AMT6</td>
      <td>PAY_AMT1</td>
      <td>PAY_AMT2</td>
      <td>PAY_AMT3</td>
      <td>PAY_AMT4</td>
      <td>PAY_AMT5</td>
      <td>PAY_AMT6</td>
      <td>default payment next month</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 25 columns</p>
</div>




```python
#drop row 203 containing original column headings
df.drop(203, inplace=True)
```


```python
len(df)
```




    30000




```python
#only row 201 should appear b/c others were dropped above
df.loc[201:210].to_markdown 
```




    <bound method DataFrame.to_markdown of 0     ID LIMIT_BAL     SEX        EDUCATION MARRIAGE AGE PAY_0 PAY_2 PAY_3  \
    201  201    180000  female  graduate school        1  38    -2    -2    -2   
    
    0   PAY_4  ... BILL_AMT4 BILL_AMT5 BILL_AMT6 PAY_AMT1 PAY_AMT2 PAY_AMT3  \
    201    -2  ...         0         0         0        0        0        0   
    
    0   PAY_AMT4 PAY_AMT5 PAY_AMT6 default payment next month  
    201        0        0        0                not default  
    
    [1 rows x 25 columns]>



*Notes*
* deleting column headings row and duplicate rows took many tries.
    * First problem, I didn't know that dropping rows doesn't stick unless you add an inplace = True parameter to your code. 
    * Second problem, I didn't realize that after you drop an index from the df (and stick it) that the index number dropped no longer appears in the df (e.g. if you drop index 0, when you use df.head() the first entry will be index 1). In other words, the data from below does not move up and fill the dropped indices as it would in an Excel file. 
* After I noticed the drops weren't sticking, I tried to solve the problem by naming a new df and setting that = to the df.drop_duplicated. That was unsuccessful. 
* Lavana advised me to add an 'inplace=True' parameter to all my drop commands. [this](https://stackabuse.com/python-with-pandas-dataframe-tutorial-with-examples/) helped me understand why that parameter is needed.
* I found this code on a tutorial to find non-numeric data in a column ```non_numeric_limit = df.LIMIT_BAL.str.contains('[^0-9.-]') / df.loc[non_numeric_limit].head()```
---

## Re-set Data Types of Variables


```python
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
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>120000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>90000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>50000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>50000</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>not default</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
#look at data types for each variable
df.dtypes
```




    0
    ID                            object
    LIMIT_BAL                     object
    SEX                           object
    EDUCATION                     object
    MARRIAGE                      object
    AGE                           object
    PAY_0                         object
    PAY_2                         object
    PAY_3                         object
    PAY_4                         object
    PAY_5                         object
    PAY_6                         object
    BILL_AMT1                     object
    BILL_AMT2                     object
    BILL_AMT3                     object
    BILL_AMT4                     object
    BILL_AMT5                     object
    BILL_AMT6                     object
    PAY_AMT1                      object
    PAY_AMT2                      object
    PAY_AMT3                      object
    PAY_AMT4                      object
    PAY_AMT5                      object
    PAY_AMT6                      object
    default payment next month    object
    dtype: object



*Notes*
* I thought after dropping all non-numerical values from the df, the vars with numerical values would change to int64 or float64. They did not. Ben said data types will need to be changed manually.

### Continuous Variables


```python
df.LIMIT_BAL.value_counts()
```




    50000      3365
    20000      1976
    30000      1610
    80000      1567
    200000     1528
               ... 
    740000        2
    690000        1
    327680        1
    760000        1
    1000000       1
    Name: LIMIT_BAL, Length: 81, dtype: int64




```python
non_numeric_limit = df.LIMIT_BAL.str.contains('[^0-9.-]') 
df.loc[non_numeric_limit].head()
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
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 25 columns</p>
</div>




```python
#change data type of LIMIT_BAL var from object to int
df["LIMIT_BAL"] = df['LIMIT_BAL'].astype('int')

#change data type of AGE var from object to int
df["AGE"] = df['AGE'].astype('int')
```


```python
#get number of obs for each value in PAY_0
df.PAY_0.value_counts()
```




    0     14737
    -1     5686
    1      3688
    -2     2759
    2      2667
    3       322
    4        76
    5        26
    8        19
    6        11
    7         9
    Name: PAY_0, dtype: int64




```python
#get number of obs for each value in PAY_4
df.PAY_4.value_counts()
```




    0     16455
    -1     5687
    -2     4348
    2      3159
    3       180
    4        69
    7        58
    5        35
    6         5
    1         2
    8         2
    Name: PAY_4, dtype: int64




```python
#get number of obs for each value in BILL_AM1
df.BILL_AMT1.value_counts()
```




    0         2008
    390        244
    780         76
    326         72
    316         63
              ... 
    216          1
    192727       1
    4434         1
    38579        1
    96644        1
    Name: BILL_AMT1, Length: 22723, dtype: int64




```python
#change data type of BILL_AMT1 var from object to int
df["BILL_AMT1"] = df['BILL_AMT1'].astype('int')

#change data type of BILL_AMT2 var from object to int
df["BILL_AMT2"] = df['BILL_AMT2'].astype('int')

#change data type of BILL_AMT3 var from object to int
df["BILL_AMT3"] = df['BILL_AMT3'].astype('int')

#change data type of BILL_AMT4 var from object to int
df["BILL_AMT4"] = df['BILL_AMT4'].astype('int')

#change data type of BILL_AMT5 var from object to int
df["BILL_AMT5"] = df['BILL_AMT5'].astype('int')

#change data type of BILL_AMT6 var from object to int
df["BILL_AMT6"] = df['BILL_AMT6'].astype('int')
```


```python
df.PAY_AMT4.value_counts()
```




    0        6408
    1000     1394
    2000     1214
    3000      887
    5000      810
             ... 
    12164       1
    17133       1
    2466        1
    3187        1
    3859        1
    Name: PAY_AMT4, Length: 6937, dtype: int64




```python
#change data type of PAY_AMT1 var from object to int
df["PAY_AMT1"] = df['PAY_AMT1'].astype('int')

#change data type of PAY_AMT2 var from object to int
df["PAY_AMT2"] = df['PAY_AMT2'].astype('int')

#change data type of PAY_AMT3 var from object to int
df["PAY_AMT3"] = df['PAY_AMT3'].astype('int')

#change data type of PAY_AMT4 var from object to int
df["PAY_AMT4"] = df['PAY_AMT4'].astype('int')

#change data type of PAY_AMT5 var from object to int
df["PAY_AMT5"] = df['PAY_AMT5'].astype('int')

#change data type of PAY_AMT6 var from object to int
df["PAY_AMT6"] = df['PAY_AMT6'].astype('int')

#rename default payment next month column
df.rename(columns={"default payment next month": "DEFAULT"}, inplace = True)

#verify change
df.dtypes
```




    0
    ID           object
    LIMIT_BAL     int32
    SEX          object
    EDUCATION    object
    MARRIAGE     object
    AGE           int32
    PAY_0        object
    PAY_2        object
    PAY_3        object
    PAY_4        object
    PAY_5        object
    PAY_6        object
    BILL_AMT1     int32
    BILL_AMT2     int32
    BILL_AMT3     int32
    BILL_AMT4     int32
    BILL_AMT5     int32
    BILL_AMT6     int32
    PAY_AMT1      int32
    PAY_AMT2      int32
    PAY_AMT3      int32
    PAY_AMT4      int32
    PAY_AMT5      int32
    PAY_AMT6      int32
    DEFAULT      object
    dtype: object



### Categorical Variables


```python
#get counts of male and female customers in df
df.SEX.value_counts()
```




    female    18112
    male      11888
    Name: SEX, dtype: int64




```python
#get counts for education categories
df.EDUCATION.value_counts()
```




    university         14030
    graduate school    10585
    high school         4917
    other                468
    Name: EDUCATION, dtype: int64




```python
#get counts for marriage categories
df.MARRIAGE.value_counts()
```




    2    15964
    1    13659
    3      323
    0       54
    Name: MARRIAGE, dtype: int64




```python
#get counts for categories of the pay_0 var
df.PAY_0.value_counts()
```




    0     14737
    -1     5686
    1      3688
    -2     2759
    2      2667
    3       322
    4        76
    5        26
    8        19
    6        11
    7         9
    Name: PAY_0, dtype: int64




```python
# get counts for the target var, i.e. DEFAULT
df.DEFAULT.value_counts()
```




    not default    23364
    default         6636
    Name: DEFAULT, dtype: int64



## Rename Features


```python
#rename PAY features to convey month
df.rename(columns = {'PAY_0': 'pay_s', 'PAY_2':'pay_ag', 'PAY_3':'pay_jy','PAY_4':'pay_ju', 'PAY_5':'pay_m', 'PAY_6':'pay_ap'}, inplace = True)

#rename BILL_AMT features to convey month
df.rename(columns = {'BILL_AMT1': 'bill_s', 'BILL_AMT2':'bill_ag', 'BILL_AMT3':'bill_jy','BILL_AMT4':'bill_ju', 'BILL_AMT5':'bill_m', 'BILL_AMT6':'bill_ap'}, inplace = True)

#rename PAY_AMT features to convey month
df.rename(columns = {'PAY_AMT1': 'pmt_s', 'PAY_AMT2':'pmt_ag', 'PAY_AMT3':'pmt_jy','PAY_AMT4':'pmt_ju', 'PAY_AMT5':'pmt_m', 'PAY_AMT6':'pmt_ap'}, inplace = True)

#verify features were renamed
df.head().to_markdown
```




    <bound method DataFrame.to_markdown of 0 ID  LIMIT_BAL     SEX   EDUCATION MARRIAGE  AGE pay_s pay_ag pay_jy pay_ju  \
    1  1      20000  female  university        1   24     2      2     -1     -1   
    2  2     120000  female  university        2   26    -1      2      0      0   
    3  3      90000  female  university        2   34     0      0      0      0   
    4  4      50000  female  university        1   37     0      0      0      0   
    5  5      50000    male  university        1   57    -1      0     -1      0   
    
    0  ... bill_ju bill_m  bill_ap  pmt_s  pmt_ag  pmt_jy  pmt_ju  pmt_m  pmt_ap  \
    1  ...       0      0        0      0     689       0       0      0       0   
    2  ...    3272   3455     3261      0    1000    1000    1000      0    2000   
    3  ...   14331  14948    15549   1518    1500    1000    1000   1000    5000   
    4  ...   28314  28959    29547   2000    2019    1200    1100   1069    1000   
    5  ...   20940  19146    19131   2000   36681   10000    9000    689     679   
    
    0      DEFAULT  
    1      default  
    2      default  
    3  not default  
    4  not default  
    5  not default  
    
    [5 rows x 25 columns]>



## Reorder Features


```python
#define a new fuction to move and re-order columns in a df (code lifted from toward data science post)
def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])
```


```python
#move LIMIT_BAL column next to other account/credit variables
df = movecol(df, 
             cols_to_move=['LIMIT_BAL'], 
             ref_col='AGE',
             place='After')
df
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
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>LIMIT_BAL</th>
      <th>pay_s</th>
      <th>pay_ag</th>
      <th>pay_jy</th>
      <th>pay_ju</th>
      <th>...</th>
      <th>bill_ju</th>
      <th>bill_m</th>
      <th>bill_ap</th>
      <th>pmt_s</th>
      <th>pmt_ag</th>
      <th>pmt_jy</th>
      <th>pmt_ju</th>
      <th>pmt_m</th>
      <th>pmt_ap</th>
      <th>DEFAULT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>120000</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>90000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>57</td>
      <td>50000</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30199</th>
      <td>29996</td>
      <td>male</td>
      <td>high school</td>
      <td>1</td>
      <td>39</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>88004</td>
      <td>31237</td>
      <td>15980</td>
      <td>8500</td>
      <td>20000</td>
      <td>5003</td>
      <td>3047</td>
      <td>5000</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>30200</th>
      <td>29997</td>
      <td>male</td>
      <td>high school</td>
      <td>2</td>
      <td>43</td>
      <td>150000</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>8979</td>
      <td>5190</td>
      <td>0</td>
      <td>1837</td>
      <td>3526</td>
      <td>8998</td>
      <td>129</td>
      <td>0</td>
      <td>0</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>30201</th>
      <td>29998</td>
      <td>male</td>
      <td>university</td>
      <td>2</td>
      <td>37</td>
      <td>30000</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>-1</td>
      <td>...</td>
      <td>20878</td>
      <td>20582</td>
      <td>19357</td>
      <td>0</td>
      <td>0</td>
      <td>22000</td>
      <td>4200</td>
      <td>2000</td>
      <td>3100</td>
      <td>default</td>
    </tr>
    <tr>
      <th>30202</th>
      <td>29999</td>
      <td>male</td>
      <td>high school</td>
      <td>1</td>
      <td>41</td>
      <td>80000</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>52774</td>
      <td>11855</td>
      <td>48944</td>
      <td>85900</td>
      <td>3409</td>
      <td>1178</td>
      <td>1926</td>
      <td>52964</td>
      <td>1804</td>
      <td>default</td>
    </tr>
    <tr>
      <th>30203</th>
      <td>30000</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>46</td>
      <td>50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>36535</td>
      <td>32428</td>
      <td>15313</td>
      <td>2078</td>
      <td>1800</td>
      <td>1430</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>default</td>
    </tr>
  </tbody>
</table>
<p>30000 rows × 25 columns</p>
</div>




```python
#re-order pay columns
df = movecol(df, 
             cols_to_move=['pay_ap', 'pay_m', 'pay_ju', 'pay_jy', 'pay_ag', 'pay_s'], 
             ref_col='LIMIT_BAL',
             place='After')
df
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
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>LIMIT_BAL</th>
      <th>pay_ap</th>
      <th>pay_m</th>
      <th>pay_ju</th>
      <th>pay_jy</th>
      <th>...</th>
      <th>bill_ju</th>
      <th>bill_m</th>
      <th>bill_ap</th>
      <th>pmt_s</th>
      <th>pmt_ag</th>
      <th>pmt_jy</th>
      <th>pmt_ju</th>
      <th>pmt_m</th>
      <th>pmt_ap</th>
      <th>DEFAULT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>20000</td>
      <td>-2</td>
      <td>-2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>120000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>90000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>57</td>
      <td>50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30199</th>
      <td>29996</td>
      <td>male</td>
      <td>high school</td>
      <td>1</td>
      <td>39</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>88004</td>
      <td>31237</td>
      <td>15980</td>
      <td>8500</td>
      <td>20000</td>
      <td>5003</td>
      <td>3047</td>
      <td>5000</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>30200</th>
      <td>29997</td>
      <td>male</td>
      <td>high school</td>
      <td>2</td>
      <td>43</td>
      <td>150000</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>8979</td>
      <td>5190</td>
      <td>0</td>
      <td>1837</td>
      <td>3526</td>
      <td>8998</td>
      <td>129</td>
      <td>0</td>
      <td>0</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>30201</th>
      <td>29998</td>
      <td>male</td>
      <td>university</td>
      <td>2</td>
      <td>37</td>
      <td>30000</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>2</td>
      <td>...</td>
      <td>20878</td>
      <td>20582</td>
      <td>19357</td>
      <td>0</td>
      <td>0</td>
      <td>22000</td>
      <td>4200</td>
      <td>2000</td>
      <td>3100</td>
      <td>default</td>
    </tr>
    <tr>
      <th>30202</th>
      <td>29999</td>
      <td>male</td>
      <td>high school</td>
      <td>1</td>
      <td>41</td>
      <td>80000</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>52774</td>
      <td>11855</td>
      <td>48944</td>
      <td>85900</td>
      <td>3409</td>
      <td>1178</td>
      <td>1926</td>
      <td>52964</td>
      <td>1804</td>
      <td>default</td>
    </tr>
    <tr>
      <th>30203</th>
      <td>30000</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>46</td>
      <td>50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>36535</td>
      <td>32428</td>
      <td>15313</td>
      <td>2078</td>
      <td>1800</td>
      <td>1430</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>default</td>
    </tr>
  </tbody>
</table>
<p>30000 rows × 25 columns</p>
</div>




```python
#re-order pmt columns
df = movecol(df, 
             cols_to_move=['pmt_ap', 'pmt_m', 'pmt_ju', 'pmt_jy', 'pmt_ag', 'pmt_s'], 
             ref_col='LIMIT_BAL',
             place='After')
df
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
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>LIMIT_BAL</th>
      <th>pmt_ap</th>
      <th>pmt_m</th>
      <th>pmt_ju</th>
      <th>pmt_jy</th>
      <th>...</th>
      <th>pay_jy</th>
      <th>pay_ag</th>
      <th>pay_s</th>
      <th>bill_s</th>
      <th>bill_ag</th>
      <th>bill_jy</th>
      <th>bill_ju</th>
      <th>bill_m</th>
      <th>bill_ap</th>
      <th>DEFAULT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>20000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-1</td>
      <td>2</td>
      <td>2</td>
      <td>3913</td>
      <td>3102</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>120000</td>
      <td>2000</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>-1</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>90000</td>
      <td>5000</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239</td>
      <td>14027</td>
      <td>13559</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>50000</td>
      <td>1000</td>
      <td>1069</td>
      <td>1100</td>
      <td>1200</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990</td>
      <td>48233</td>
      <td>49291</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>57</td>
      <td>50000</td>
      <td>679</td>
      <td>689</td>
      <td>9000</td>
      <td>10000</td>
      <td>...</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>8617</td>
      <td>5670</td>
      <td>35835</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30199</th>
      <td>29996</td>
      <td>male</td>
      <td>high school</td>
      <td>1</td>
      <td>39</td>
      <td>220000</td>
      <td>1000</td>
      <td>5000</td>
      <td>3047</td>
      <td>5003</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>188948</td>
      <td>192815</td>
      <td>208365</td>
      <td>88004</td>
      <td>31237</td>
      <td>15980</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>30200</th>
      <td>29997</td>
      <td>male</td>
      <td>high school</td>
      <td>2</td>
      <td>43</td>
      <td>150000</td>
      <td>0</td>
      <td>0</td>
      <td>129</td>
      <td>8998</td>
      <td>...</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1683</td>
      <td>1828</td>
      <td>3502</td>
      <td>8979</td>
      <td>5190</td>
      <td>0</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>30201</th>
      <td>29998</td>
      <td>male</td>
      <td>university</td>
      <td>2</td>
      <td>37</td>
      <td>30000</td>
      <td>3100</td>
      <td>2000</td>
      <td>4200</td>
      <td>22000</td>
      <td>...</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3565</td>
      <td>3356</td>
      <td>2758</td>
      <td>20878</td>
      <td>20582</td>
      <td>19357</td>
      <td>default</td>
    </tr>
    <tr>
      <th>30202</th>
      <td>29999</td>
      <td>male</td>
      <td>high school</td>
      <td>1</td>
      <td>41</td>
      <td>80000</td>
      <td>1804</td>
      <td>52964</td>
      <td>1926</td>
      <td>1178</td>
      <td>...</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>-1645</td>
      <td>78379</td>
      <td>76304</td>
      <td>52774</td>
      <td>11855</td>
      <td>48944</td>
      <td>default</td>
    </tr>
    <tr>
      <th>30203</th>
      <td>30000</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>46</td>
      <td>50000</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>1430</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47929</td>
      <td>48905</td>
      <td>49764</td>
      <td>36535</td>
      <td>32428</td>
      <td>15313</td>
      <td>default</td>
    </tr>
  </tbody>
</table>
<p>30000 rows × 25 columns</p>
</div>




```python
#re-order bill columns
df = movecol(df, 
             cols_to_move=['bill_ap', 'bill_m', 'bill_ju', 'bill_jy', 'bill_ag', 'bill_s'], 
             ref_col='LIMIT_BAL',
             place='After')

df.head().to_markdown
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-834561f03928> in <module>
          1 #re-order bill columns
    ----> 2 df = movecol(df, 
          3              cols_to_move=['bill_ap', 'bill_m', 'bill_ju', 'bill_jy', 'bill_ag', 'bill_s'],
          4              ref_col='LIMIT_BAL',
          5              place='After')
    

    NameError: name 'df' is not defined


## Export Cleaned Data


```python
#export cleaned transformed data as .csv file
df.to_csv('cleaned_creditone_data.csv', index = False, header=True)
```
