from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#connect python to MySQL server (database?)
db_connection_str = 'mysql+pymysql://deepanalytics:Sqltask1234get_ipython().getoutput("@34.73.222.197/deepanalytics'")


#create engine for SQLalchemy to interface with the database API (see notes below)
db_connection = create_engine(db_connection_str)

# first time successfully queried db without this line, so not entirely
#sure why it is needed. asked on slack, answer = link to oriely article



#query the credit one data and extract it into a pd df
sql_query = pd.read_sql('SELECT * FROM credit', con=db_connection)


#create pandas df from SQL query
df = pd.DataFrame(sql_query)


df.head().to_markdown


df.info()


#export df as .csv file
df.to_csv (r'C:\Users\Kpiat\export_data.csv', index = False)


#import exported csv file with raw data
df = pd.read_csv('raw_credit_one_data.csv')


#verify imported data looks as expected 
df.head()


#look at number of rows and columns in df as benchmark before making any changes 
df.shape


#change column headings to values in index (row) 0 
df.columns = df.iloc[0]


#verify change was successful
df.head().to_markdown


#drop row 0 that contains column headings
df.drop(df.index[0], inplace = True)


#verify df is -1 row
len(df)


#verify column headings row was dropped
df.head()


#examine rows 200-206 of df
df.loc[200:206]


#find and print all duplicate rows in the df
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)


#drop duplicate rows from the df
df.drop_duplicates(inplace = True)  #inplace=True is to make change to df permanent


#verify rows were dropped
len(df)


#check for missing values
print(df.isnull().sum())


#drop the row with missing ID
df.dropna(inplace=True)


len(df)


df.head()


#locate rows 201-206 (not inclusive)
df.loc[201:206]


#drop row 203 containing original column headings
df.drop(203, inplace=True)


len(df)


#only row 201 should appear b/c others were dropped above
df.loc[201:210].to_markdown 


df.head()


#look at data types for each variable
df.dtypes


df.LIMIT_BAL.value_counts()


non_numeric_limit = df.LIMIT_BAL.str.contains('[^0-9.-]') 
df.loc[non_numeric_limit].head()


#change data type of LIMIT_BAL var from object to int
df["LIMIT_BAL"] = df['LIMIT_BAL'].astype('int')

#change data type of AGE var from object to int
df["AGE"] = df['AGE'].astype('int')


#get number of obs for each value in PAY_0
df.PAY_0.value_counts()


#get number of obs for each value in PAY_4
df.PAY_4.value_counts()


#get number of obs for each value in BILL_AM1
df.BILL_AMT1.value_counts()


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


df.PAY_AMT4.value_counts()


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


#get counts of male and female customers in df
df.SEX.value_counts()


#get counts for education categories
df.EDUCATION.value_counts()


#get counts for marriage categories
df.MARRIAGE.value_counts()


#get counts for categories of the pay_0 var
df.PAY_0.value_counts()


# get counts for the target var, i.e. DEFAULT
df.DEFAULT.value_counts()


#rename PAY features to convey month
df.rename(columns = {'PAY_0': 'pay_s', 'PAY_2':'pay_ag', 'PAY_3':'pay_jy','PAY_4':'pay_ju', 'PAY_5':'pay_m', 'PAY_6':'pay_ap'}, inplace = True)

#rename BILL_AMT features to convey month
df.rename(columns = {'BILL_AMT1': 'bill_s', 'BILL_AMT2':'bill_ag', 'BILL_AMT3':'bill_jy','BILL_AMT4':'bill_ju', 'BILL_AMT5':'bill_m', 'BILL_AMT6':'bill_ap'}, inplace = True)

#rename PAY_AMT features to convey month
df.rename(columns = {'PAY_AMT1': 'pmt_s', 'PAY_AMT2':'pmt_ag', 'PAY_AMT3':'pmt_jy','PAY_AMT4':'pmt_ju', 'PAY_AMT5':'pmt_m', 'PAY_AMT6':'pmt_ap'}, inplace = True)

#verify features were renamed
df.head().to_markdown


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


#move LIMIT_BAL column next to other account/credit variables
df = movecol(df, 
             cols_to_move=['LIMIT_BAL'], 
             ref_col='AGE',
             place='After')
df


#re-order pay columns
df = movecol(df, 
             cols_to_move=['pay_ap', 'pay_m', 'pay_ju', 'pay_jy', 'pay_ag', 'pay_s'], 
             ref_col='LIMIT_BAL',
             place='After')
df


#re-order pmt columns
df = movecol(df, 
             cols_to_move=['pmt_ap', 'pmt_m', 'pmt_ju', 'pmt_jy', 'pmt_ag', 'pmt_s'], 
             ref_col='LIMIT_BAL',
             place='After')
df


#re-order bill columns
df = movecol(df, 
             cols_to_move=['bill_ap', 'bill_m', 'bill_ju', 'bill_jy', 'bill_ag', 'bill_s'], 
             ref_col='LIMIT_BAL',
             place='After')

df.head().to_markdown


#export cleaned transformed data as .csv file
df.to_csv('cleaned_creditone_data.csv', index = False, header=True)
