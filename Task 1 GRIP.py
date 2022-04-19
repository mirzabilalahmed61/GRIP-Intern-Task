#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Spark Foundation
# 

# # Data Science And Business Analytics Intern
# 

# # Author :  Mirza Bilal Ahmed

# # Task 1 :  Prediction Using  Supervised ML

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data import successfully")
data.head(10)


# In[4]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# # Data Visualization 

# In[16]:


data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Linear Regression Model

# In[17]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.80,test_size=0.20,random_state=42)


# 
# # Training The Model

# In[18]:


from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(X_train, y_train)
y_predict= linearRegressor.predict(X_train)


# # Training The Alogirthm
# 
# Now Split the data into training an test set 

# In[19]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[20]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.show()


# # Checking the Accuracy of Training and Testing set

# In[21]:


print('Test Score')
print(regressor.score(X_test, y_test))
print('Training Score')
print(regressor.score(X_train, y_train))


# In[22]:


y_test


# In[23]:


y_predict


# In[24]:


y_predict[:5]


# In[25]:


data= pd.DataFrame({'Actual': y_test,'Predicted': y_predict[:5]})
data


# In[26]:


print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))


# # Model Evalution Metrics

# In[27]:


mean_squ_error = mean_squared_error(y_test, y_predict[:5])
mean_abs_error = mean_absolute_error(y_test, y_predict[:5])
print("Mean Squred Error:",mean_squ_error)
print("Mean absolute Error:",mean_abs_error)


# In[ ]:




