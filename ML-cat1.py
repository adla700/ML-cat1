#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


#generate random data for regression
x = (np.random.rand(1000)*100)
x= np.asarray(x,dtype='int')
y = x+np.random.normal(0,50,1000)
y= np.asarray(y,dtype='int')


# In[3]:


print(x)


# In[4]:


print(y)


# In[5]:


inds = np.random.permutation(len(x))
size = round(len(x)*0.7)
train,test = inds[:size],inds[size:]
print(inds[:size])
x_train,x_test,y_train,y_test = x[train],x[test],y[train],y[test]


# In[6]:


#build simple linear reg
mean_x = np.mean(x_train)
mean_y = np.mean(y_train)


# In[7]:


m = len(x_train)
numer = 0
denom = 0
for i in range(m):
  numer += (x_train[i] - mean_x) * (y_train[i] - mean_y)
  denom += (x_train[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)   #c= ymean -(m * xmean)
print(m)
print(c)


# In[8]:


max_x = np.max(x_train) + 100
min_x = np.min(y_train) - 100

x = np.linspace (min_x, max_x, 100)   
y = c + m * x

plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(x, y, c='#ef5423', label='data points')

plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[9]:


print(type(x))


# In[10]:


#calculating mean absolute error
sum1=0
diff=0
# m is total no.of values --> m=len(x)
for i in range(int(m)):
    y_pred = c + (m * x[i])
    diff=abs(y_pred-y[i])
    sum1+=diff
total=sum1/m
total


# In[14]:


#calculating rmse
import math
sum1=0
diff1=0
for i in range(int(m)):
    y_pred = c + m * x[i]
    diff1 = (y_pred-y[i]) ** 2
    sum1 += diff1
    
total1=math.sqrt(sum1/m)
total1


# In[15]:


#use gradient descent
import math 
import matplotlib.pyplot as plt 


# In[16]:


x=pd.DataFrame(x)
y=pd.DataFrame(y)
data=pd.DataFrame()
data['x']=x
data['y']=y


# In[17]:


data


# In[18]:


m_gd,c_gd = 0,0
l = 0.0001
epochs = 20000
n = float(len(x_train))
for i in range(epochs): 
    y_p= m_gd*x_train + c_gd
    D_m = (-2/n) * sum(x_train * (y_train - y_p))
    D_c = (-2/n) * sum(y_train - y_p)
    m_gd = m_gd - l * D_m
    c_gd = c_gd - l * D_c


# In[19]:


m_gd,c_gd


# In[20]:


y_pred_gd = m_gd*x_test + c_gd


# In[22]:


#using SKLEARN
import sklearn
data.plot.scatter(x='x', y='y', title='Scatterplot of hours and scores percentages');
print(data.corr())


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[24]:


SEED = 42


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = SEED)


# In[26]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[27]:


regressor.fit(X_train, y_train)


# In[28]:


print(regressor.intercept_)


# In[29]:


print(regressor.coef_)


# In[30]:


y_pred = regressor.predict(X_test)


# In[31]:


df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)


# In[32]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[33]:


import numpy as np


# In[34]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


# In[35]:


print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


# In[ ]:




