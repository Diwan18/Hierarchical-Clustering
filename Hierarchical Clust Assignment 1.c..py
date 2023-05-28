# -*- coding: utf-8 -*-
"""
Created on Sat May  6 20:39:16 2023

@author: Admin
"""
"""
⦁	Perform clustering analysis on the telecom data set. 
The data is a mixture of both categorical and numerical data. 
It consists of the number of customers who churn out. 
Derive insights and get possible information on factors that may affect the churn decision.
 Refer to Telco_customer_churn.xlsx dataset.


"""

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

TelecomDataset = pd.read_excel(
    r"D:\Hierarchical Assignments Dataset\Datasets Assignment 1\Telco_customer_churn.xlsx")

TelecomDataset.head()

TelecomDataset.describe()
TelecomDataset.info()
TelecomDataset.columns.shape

TelecomDataset.isna().sum()

Telecom1 = TelecomDataset.duplicated().sum()

# Data Preprocessing

# Label Encoding
# we have most of the data in the form of category which needs to be changed to Numerical Data.

#Hence we do label encoding for the features to encode the labesl within the features

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

x = TelecomDataset.iloc[:, [3,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23]]
x.isna().sum()       
Numerical_data = TelecomDataset.iloc[:, [0,1,2,4,5,8,12,24,25,26,27,28,29]]     


x['Referred a Friend']=labelencoder.fit_transform(x['Referred a Friend'])
x['Offer']=labelencoder.fit_transform(x['Offer'])
x['Phone Service']=labelencoder.fit_transform(x['Phone Service'])
x['Multiple Lines']=labelencoder.fit_transform(x['Multiple Lines'])
x['Internet Service']=labelencoder.fit_transform(x['Internet Service'])
x['Internet Type']=labelencoder.fit_transform(x['Internet Type'])
x['Online Backup']=labelencoder.fit_transform(x['Online Backup'])
x['Online Security']=labelencoder.fit_transform(x['Online Security'])
x['Device Protection Plan']=labelencoder.fit_transform(x['Device Protection Plan'])
x['Premium Tech Support']=labelencoder.fit_transform(x['Premium Tech Support'])
x['Streaming TV']=labelencoder.fit_transform(x['Streaming TV'])
x['Streaming Movies']=labelencoder.fit_transform(x['Streaming Movies'])
x['Streaming Music']=labelencoder.fit_transform(x['Streaming Music'])
x['Unlimited Data']=labelencoder.fit_transform(x['Unlimited Data'])
x['Contract']=labelencoder.fit_transform(x['Contract'])
x['Paperless Billing']=labelencoder.fit_transform(x['Paperless Billing'])
x['Payment Method']=labelencoder.fit_transform(x['Payment Method'])


Numerical_data = pd.DataFrame(Numerical_data)

# concatenate x and y
telecom_new=pd.concat([x,Numerical_data ],axis=1)
telecom_new.columns
telecom_new.isna().sum()
telecom_new.describe().transpose()
telecom_new.info()


telecom_new.drop(['Customer ID' , 'Quarter' ], axis = 1, inplace = True)

# Checking for Outliers 
plt.boxplot(telecom_new)

from AutoClean import AutoClean
clean_pipeline = AutoClean(telecom_new, mode = 'manual', missing_num = 'auto',
                           outliers = 'winz', encode_categ = 'auto')


Telecom_clean = clean_pipeline.output
plt.boxplot(Telecom_clean)

# ## Normalization/MinMax Scaler - To address the scale differences

# ### Python Pipelines
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

Telecom_clean.info()

cols = list(Telecom_clean.columns)

print(cols)

pipe1 = make_pipeline(MinMaxScaler())

# Train the data preprocessing pipeline on data
Telecom_pipelined = pd.DataFrame(pipe1.fit_transform(Telecom_clean), columns = cols, index = Telecom_clean.index)
Telecom_pipelined.head()
Telecom_pipelined.describe().transpose() # scale is normalized to min = 0; max = 1
###### End of Data Preprocessing ######


# # CLUSTERING MODEL BUILDING

# ### Hierarchical Clustering - Agglomerative Clustering

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering 
import matplotlib.pyplot as plt

#plt.figure( figsize = (15, 8))

#plt.title('Hierarchical Clustering Dendrogram')

#plt.xlabel('Index')

#plt.ylabel('Euclidean distance')

#plt.show(dendrogram(linkage(Telecom_pipelined, method  = "complete")))
#tree_plot = dendrogram(linkage(Telecom_pipelined, method  = "ward"))



plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')

plt.show(dendrogram(linkage(Telecom_pipelined, method  = "ward")))


# Experiment to obtain the best clusters by altering the parameters

# ## Cluster Evaluation Library

# pip install clusteval
# Refer to link: https://pypi.org/project/clusteval

from clusteval import clusteval
import numpy as np

# Silhouette cluster evaluation. 
ce = clusteval(evaluate = 'silhouette')

df_array = np.array(Telecom_clean)

# Fit
ce.fit(df_array)

# Plot
ce.plot()


## Using the report from clusteval library building 2 clusters
# Fit using agglomerativeClustering with metrics: euclidean, and linkage: ward

Telecom_2clust = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')

y_Telecom_2clust = Telecom_2clust.fit_predict(Telecom_clean)

# Cluster labels
Telecom_2clust.labels_

cluster_labels2 = pd.Series(Telecom_2clust.labels_) 

from sklearn import metrics
metrics.silhouette_score(Telecom_clean, cluster_labels2)

'''Alternatively, we can use:'''
# **Calinski Harabasz:**

metrics.calinski_harabasz_score(Telecom_clean, cluster_labels2)

# **Davies-Bouldin Index:**

metrics.davies_bouldin_score(Telecom_clean, cluster_labels2)


# Concate the Results with data
df_2clust = pd.concat([cluster_labels2, TelecomDataset], axis = 1)

df_2clust = df_2clust.rename(columns = {0:'clusters'})
df_2clust.head()

# Aggregate using the mean of each cluster
df_2clust.iloc[:, :].groupby(df_2clust.clusters).mean().transpose()


# Save the Results to a CSV file
df_3clust = pd.concat([ TelecomDataset, cluster_labels2], axis = 1)
df_3clust = df_3clust.rename(columns = {0:'clusters'})
df_3clust['clusters'] = df_3clust['clusters'].replace([0, 1], ['Least possibility to churn', 'High Possibilty churn '])
df_3clust.to_csv('telecom.csv', encoding = 'utf-8')


"""
Cluster 1 = These are the customers that are frequent users and take up the offers as well, 
Internet used by them is moderate to heavy, but take on offers as well. 
The revenue earned through these is also the best. 
Hence, these are the customers that are least likely to churn.

Cluster 2 = These are the customers that are least on all the criterias whether being using the
 services, to net usage, and dont take up the offers as well, additionally the revenue
 earned through them is the least. Hence we infer that these are the ones that churn the most.
"""

import os
os.getcwd()















