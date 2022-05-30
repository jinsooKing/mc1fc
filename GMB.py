#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import GradientBoostingRegressor

import seaborn as sns



data = pd.read_csv("student-mat.csv", sep=";")
data2 = pd.read_csv("student-por.csv", sep=";")


# In[2]:


#주어진 데이터에서 사용할 데이터 추출

data3 = pd.concat([data, data2], ignore_index = True)

df = data3[['age','absences','sex','traveltime','studytime','schoolsup','freetime','G1','G2','G3']]

df


# In[3]:


for col in['sex','traveltime','studytime','schoolsup','freetime']:
    df[col] = df[col].astype('category')

df.info()


# In[4]:


# traing set와 test set을 생성

x = df[['age', 'absences','sex','traveltime','studytime','schoolsup','freetime','G1','G2']]

y = df[['G3']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)


# In[5]:


categorical = list(x_train.select_dtypes('category').columns)
print(f"Categorical columns are: {categorical}")

numerical = list(x_train.select_dtypes('number').columns)
print(f"Numerical columns are: {numerical}")


# In[6]:


# One Hot encoding of category dataset

cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown="ignore", sparse=False))
])


# In[7]:


# Scaling of Numerical dataset

from sklearn.preprocessing import Normalizer

num_pipe = Pipeline([('scaler', Normalizer())])

# Combining of both cat & num

preprocessor = ColumnTransformer([
    ('cat', cat_pipe, categorical),
    ('num', num_pipe, numerical)
])

#Fixing Pipeline on Linner regression

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipe.fit(x_train, y_train)


# In[8]:


# Predicting training data

y_train_predict = pipe.predict(x_train)
print(f"Predictions on training data: {y_train_predict}")


# In[9]:


# Predict test data

y_test_predict = pipe.predict(x_test)
print(f"Predictions on test data: {y_test_predict}")


# In[10]:


# Check r square value

r2 = r2_score(y_test, y_test_predict)
print('r2 score for a model is', r2)

RMSE = mean_squared_error(y_test,y_test_predict)
RMSE = np.sqrt(RMSE)
print('RMSE for a model is', RMSE)


# In[11]:


#Outlier removal (이상치 데이터 삭제) Q3 - Q1 사용

result = df.select_dtypes(include='number')#selecting dtypes in dataset

for i in result.columns:
    percentile25 = df[i].quantile(0.25)
    percentile75 = df[i].quantile(0.75)
    
    iqr = percentile75-percentile25
    
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    
    df[df[i] > upper_limit]
    df[df[i] < lower_limit]
    
    df_new = df[df[i] < upper_limit ]
    df_new = df[df[i] > lower_limit ]


# In[12]:


# Random Forest를 위한 새로운 dataframe

df_rf=df_new.copy(deep=True)


# In[13]:


# traing set와 test set을 생성

x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(df_rf.drop(columns=[]),
                                                            df_rf['G3'], test_size=.2, random_state=10)


# In[14]:


categorical_n = list(x_train_n.select_dtypes('category').columns)
print(f"Categorical columns are : {categorical_n}")

numerical_n = list(x_train_n.select_dtypes('number').columns)
print(f"Numerical columns are : {numerical_n}" )


# In[15]:


# One Hot Encoding

cat_pipe_n = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


# In[16]:


# Normalization이랑 categorical & numerical combine 하기


num_pipe_n = Pipeline([('scaler', Normalizer())])

preprocessor_n = ColumnTransformer([
    ('cat', cat_pipe_n, categorical_n),
    ('num', num_pipe_n, numerical_n)
])


# In[17]:


# 전처리 작업 마무리

pipe_n = Pipeline([
    ('preprocessor', preprocessor_n),
    ('model', GradientBoostingRegressor(n_estimators=400, max_depth=5, random_state=0, learning_rate=0.2))])
pipe_n.fit(x_train_n, y_train_n)


# In[18]:


# predicting training data

y_train_pred_n = pipe_n.predict(x_train_n)
print(f"Predictions on training data: {y_train_pred_n}")


# In[19]:


# predicting test data

y_test_pred_n = pipe_n.predict(x_test_n)
print(f"Predictions on test data: {y_test_pred_n}")


# In[20]:


# r2 score와 RMSE

r2_n = r2_score(y_test_n, y_test_pred_n)
print('r2_n score for a  model is', r2_n)

RMSE_n = mean_squared_error(y_test_n,y_test_pred_n)
RMSE_n = np.sqrt(RMSE_n)
print('RMSE for a model is', RMSE_n)


# In[ ]:




