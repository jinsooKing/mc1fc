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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
import warnings
warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from bayes_opt import BayesianOptimization
import xgboost
from sklearn.inspection import permutation_importance
import eli5
from eli5.sklearn import PermutationImportance
import shap

data = pd.read_csv("student-mat.csv", sep=";")

data2 = pd.read_csv("student-por.csv", sep=";")


# In[2]:


#주어진 데이터에서 사용할 데이터 추출

data3 = pd.concat([data, data2], ignore_index = True)

df = data3[['age','absences','sex','traveltime','studytime','schoolsup','freetime','G1','G2','G3']]

df


# In[43]:


for col in['sex','traveltime','studytime','schoolsup','freetime']:
    df[col] = df[col].astype('category')

df.info()


# In[4]:


# Correlation between different numerical varriable

print(df.corr())

# plotting correlation heatmap
fig, ax = plt.subplots(figsize=(20,5)) 
mask = np.triu(np.ones_like(df.corr()))
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True,mask=mask,annot_kws={'size': 13},ax=ax)
  
# displaying heatmap
plt.show()


# In[5]:


sns.boxplot(data=df, x="schoolsup", y="G1")


# In[6]:


sns.boxplot(data=df, x="schoolsup", y="G2")


# In[7]:


sns.boxplot(data=df, x="schoolsup", y="G3")


# In[8]:


# Interrelationship Between Different Grading Scheme

fig, ax =plt.subplots(1,3,figsize=(20,6), dpi=80, facecolor='w', edgecolor='k')
sns.scatterplot(data=df, x="G1", y="G3",ax=ax[0])
sns.scatterplot(data=df, x="G2", y="G3",ax=ax[1])
sns.scatterplot(data=df, x="G1", y="G2",ax=ax[2])
sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
fig.suptitle('Relationship Between Different Grade with each other', fontsize=16)
fig.show()


# In[9]:


sns.set_theme(style="ticks")

sns.jointplot(x=df['G1'], y=df['G3'], kind="hex", color="#4CB391")
sns.jointplot(x=df['G2'], y=df['G3'], kind="hex", color="#4CB391")
sns.jointplot(x=df['G1'], y=df['G2'], kind="hex", color="#4CB391")
fig.show()


# In[10]:


sns.histplot(
    df, x="absences", y="G1",
    bins=5, discrete=(True, False),
    cbar=True, cbar_kws=dict(shrink=.75)
)


# In[11]:


sns.histplot(
    df, x="absences", y="G2",
    bins=5, discrete=(True, False),
    cbar=True, cbar_kws=dict(shrink=.75)
)


# In[12]:


sns.histplot(
    df, x="absences", y="G3",
    bins=5, discrete=(True, False),
    cbar=True, cbar_kws=dict(shrink=.75)
)


# In[13]:


sns.histplot(
    df, x="G1", hue="sex", element="step",
    stat="density", common_norm=False,bins=6
)


# In[14]:


sns.histplot(
    df, x="G2", hue="sex", element="step",
    stat="density", common_norm=False,bins=6
)


# In[15]:


sns.histplot(
    df, x="G3", hue="sex", element="step",
    stat="density", common_norm=False,bins=6
)


# In[16]:


sns.histplot(
    df, x="G1", hue="studytime", element="step",
    stat="density", common_norm=False,bins=6
)


# In[17]:


sns.histplot(
    df, x="G2", hue="studytime", element="step",
    stat="density", common_norm=False,bins=6
)


# In[18]:


sns.histplot(
    df, x="G3", hue="studytime", element="step",
    stat="density", common_norm=False,bins=6
)


# In[19]:


# traing set와 test set을 생성

x = df[['age', 'absences','sex','traveltime','studytime','schoolsup','freetime','G1','G2']]

y = df[['G3']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

categorical = list(x_train.select_dtypes('category').columns)
print(f"Categorical columns are: {categorical}")

numerical = list(x_train.select_dtypes('number').columns)
print(f"Numerical columns are: {numerical}")

# One Hot encoding of category dataset

cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown="ignore", sparse=False))
])

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


# In[20]:


# Predicting training data

y_train_predict = pipe.predict(x_train)
print(f"Predictions on training data: {y_train_predict}")


# In[21]:


# Predict test data

y_test_predict = pipe.predict(x_test)
print(f"Predictions on test data: {y_test_predict}")


# In[22]:


# Check r square value

r2 = r2_score(y_test, y_test_predict)
print('r2 score for a model is', r2)

RMSE = mean_squared_error(y_test,y_test_predict)
RMSE = np.sqrt(RMSE)
print('RMSE for a model is', RMSE)


# In[23]:


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


# In[24]:


# Random Forest를 위한 새로운 dataframe

df_rf=df_new.copy(deep=True)
df_rf2=df_new.copy(deep=True)
df_rf3=df_new.copy(deep=True)


# In[25]:


# traing set와 test set을 생성

x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(df_rf.drop(columns=[]),
                                                            df_rf['G3'], test_size=.2, random_state=10)


# In[26]:


categorical_n = list(x_train_n.select_dtypes('category').columns)
print(f"Categorical columns are : {categorical_n}")

numerical_n = list(x_train_n.select_dtypes('number').columns)
print(f"Numerical columns are : {numerical_n}" )


# In[27]:


# One Hot Encoding

cat_pipe_n = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


# In[28]:


# Normalization이랑 categorical & numerical combine 하기


num_pipe_n = Pipeline([('scaler', Normalizer())])

preprocessor_n = ColumnTransformer([
    ('cat', cat_pipe_n, categorical_n),
    ('num', num_pipe_n, numerical_n)
])


# In[29]:


# 전처리 작업 마무리

pipe_n = Pipeline([
    ('preprocessor', preprocessor_n),
    ('model', RandomForestRegressor(max_depth=10, random_state=8))])
pipe_n.fit(x_train_n, y_train_n)


# In[30]:


# predicting training data

y_train_pred_n = pipe_n.predict(x_train_n)
print(f"Predictions on training data: {y_train_pred_n}")


# In[31]:


# predicting test data

y_test_pred_n = pipe_n.predict(x_test_n)
print(f"Predictions on test data: {y_test_pred_n}")


# In[32]:


# r2 score와 RMSE

r2_n = r2_score(y_test_n, y_test_pred_n)
print('r2_n score for a  model is', r2_n)

RMSE_n = mean_squared_error(y_test_n,y_test_pred_n)
RMSE_n = np.sqrt(RMSE)
print('RMSE for a model is', RMSE_n)


# In[33]:


# XGBoost 를 위한 데이터셋 생성

x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(df_rf2.drop(columns=[]),
                                                            df_rf2['G3'], test_size=.2, random_state = 156)

# category랑 numerical 분류

categorical_b = list(x_train_b.select_dtypes('category').columns)
print(f"Categorical columns are : {categorical_b}")

numerical_b = list(x_train_b.select_dtypes('number').columns)
print(f"Numerical columns are : {numerical_b}" )

# One Hot Encoding

cat_pipe_b = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Normalization이랑 categorical & numerical combine 하기


num_pipe_b = Pipeline([('scaler', Normalizer())])

preprocessor_b = ColumnTransformer([
    ('cat', cat_pipe_b, categorical_b),
    ('num', num_pipe_b, numerical_b)
])

# 전처리 작업 마무리

pipe_b = Pipeline([
    ('preprocessor', preprocessor_b),
    ('model', XGBRegressor(n_estimators=400, learning_rate=0.08, subsample=0.76, max_depth=8, gamma=0))])
pipe_b.fit(x_train_b, y_train_b)


# In[34]:


# predicting training data

y_train_pred_b = pipe_b.predict(x_train_b)
print(f"Predictions on training data: {y_train_pred_b}")


# In[35]:


# predicting test data

y_test_pred_b = pipe_b.predict(x_test_b)
print(f"Predictions on test data: {y_test_pred_b}")


# In[36]:


# r2 score와 RMSE

r2_b = r2_score(y_test_b, y_test_pred_b)
print('r2_b score for a  model is', r2_b)

RMSE_b = mean_squared_error(y_test_b,y_test_pred_b)
RMSE_b = np.sqrt(RMSE)
print('RMSE for a model is', RMSE_b)


# In[60]:


pipe_b.steps[1][1].feature_importances_


# In[77]:


import dtreeviz.trees

feat_dict= {}
for col, val in sorted(zip(df_rf2.columns, pipe_b.steps[1][1].feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val

feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

feat_df


# In[78]:


values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
sns.barplot(y=idx,x=values).set(title='Important features to predict student performance')
plt.show()


# In[ ]:




