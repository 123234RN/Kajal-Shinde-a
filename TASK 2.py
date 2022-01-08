#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Spark Foundation
# 
# Data science and Business Analytics Internship
# 
# Author: Kajal Shinde
# 
# Task 2 : prediction using Unsupervised ML

# steps:
#     step 1- Importing the dataset
#     step 3 -Finding the optimum number of clusters
#     step 4 - Applying k means clustering on the data 
#     step 5 - Visualising the clusters

# # STEP-1 Importing the data

# In[5]:


# Importing the required liabraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head()


# In[6]:


# Finding the optimum number of clusters for k_means classification
x = iris_df.iloc[:, [0,1,2,3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                   max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# plotting the results onto a line graph,
# allowing us to observe 'The elbow'
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of square within cluster')
plt.show()


# In[9]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
               max_iter = 300,n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[10]:


# Visualising the clusters - on the first two columns
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1],
            s = 100, c = 'black', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1],
            s = 100, c = 'pink', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1],
            s = 100, c = 'orange',label = 'Iris-virginica')

# plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'green', label = 'Centroids')

plt.legend()


# In[ ]:




