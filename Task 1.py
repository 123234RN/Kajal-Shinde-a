#!/usr/bin/env python
# coding: utf-8

# GRIP : The Spark Foundation
# 

# Data science and Business Analytics Internship

# Author: Kajal Shinde

# Task 1 : prediction using supervised ML

# In[2]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Reading data from remote url

# In[11]:


#Reading data from remote link
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# Exploring data

# In[12]:


#Ploting the distribution of scores
s_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# from the graph above ,we can clearly see that there is a positive relation between the number of hours studied and percentage of scores.

# PREPARING THE DATA
# 
# The next step is to divide the data into "attributes"(inputs) and "labels"(outputs)

# In[13]:


### Independent and dependent features
X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, -1].values


# Now that we have to attribute and labels , the next step is to split this data into training and test sets . We'll do this by using Scikit-Learn'sbuilt-in-train_test_split()method

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   test_size=0.2, random_state=0)


# TRAINING AND ALGORITHM 
# We have split our data into training and testing sets , and we now finally the time to train our algorithm

# In[15]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")


# In[16]:


# Plotting the regession line 
line = regressor.coef_*X+regressor.intercept_

# plotting for the test data
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# Making Predictions
# Now that we have trained our algorithm , it's time to make some prediction.

# In[17]:


print(X_test) #Testing data - In Hours
y_pred = regressor.predict(X_test) #predicting the scores


# In[18]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df


# In[19]:


# You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[20]:


from sklearn import metrics
print('Mean Absolute Error:',
     metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




