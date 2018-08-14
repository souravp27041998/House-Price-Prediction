# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("D:\study materials\data science\data sets\kc_house_data.csv")
y=df['price']

l1=list()
l2=list()


#function for feature selection

def dependent(col_name,corr):
    for i in corr.columns:
        if abs(corr[col_name][i])>=0.15 and abs(corr[col_name][i])!=1:
            l1.append(i)
        else:
            l2.append(i)
            
corr1=df.corr()
dependent('price',corr1)

#drop the irrelevent features
df_1=df.drop(l2,axis=1)
df_1=df_1.drop(['date','sqft_living15','view','sqft_living15'],axis=1)
for var in l2:
    l2.remove(var)
l2=[]
df_1['price']=y
corr2=df_1.corr()
dependent('price',corr2)
df_1=df_1.drop(l2,axis=1)

#split the data and build model 
x=df_1.loc[:]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("accuracy of the model is:",r2_score(y_test,y_pred))


