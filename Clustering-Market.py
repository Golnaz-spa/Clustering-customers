import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
import datetime
import matplotlib.cm as cm
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from datetime import date
import re

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

"""
Data set of online store. 
Goal --> We want to cluster our customers to three levels
"""

df = pd.read_csv('E:/file/Marketnc.csv')
print(df.head(3))
print(df.shape)

#find unique values of Invoice
print(np.unique(df['InvoiceNo']))
print(len(np.unique(df['InvoiceNo'])))

#finding missing values
print(np.sum(df.isna()))

#drop missing values as we do not know how to identify customers
df.dropna(inplace=True)

#get summary of all columns
print(df.describe(include='all'))

#find type of columns
print(df.dtypes)

#there are character 'c' in soe invoicenumbers. It means that they are cancelled. So want to find cancel items and remove them from our data
#some times the quantity is negative because of canceled items
cancellations=df['InvoiceNo'].str.contains('C')
print(cancellations)
df=df.loc[cancellations!=True]

"""
Feature Engineering
we want to find the clusters of each cutomers. Find the meaning of loyalty for customers.RMF is used for defining loyalty customers.
R --> recency   -- last time that customers buy from the store
f --> frequency -- mean of the frequency that customer buy from the store
m --> monetary   -- how much money spend for buying from the store

Loyal Customer     : Recency is low, montery and frequency are high
Not Loyal Customer : Recency is high, montery and frequency are low
Neutral Customer   : Recency montery and frequency are mid
"""

#finding Recency
df['InvoiceDate']=df['InvoiceDate'].agg(pd.Timestamp)
#set 04/11/2012 for today time
today=pd.Timestamp('04/11/2012')
df['Days']=df['InvoiceDate'].agg(lambda x:(today-x).days)
Recency=df.groupby('CustomerID').Days.min()
Latency=df.groupby('CustomerID').Days.mean()

#find Frequency
Frequency=df.groupby('CustomerID').InvoiceNo.unique().agg(len)
#can use nunique() instead of unique() and len -->  Frequency=df.groupby('CustomerID').InvoiceNo.nunique()

#find Monetary
df['Cost']=df['Quantity']*df['UnitPrice']
Monetary=df.groupby('CustomerID').sum().loc[:,'Cost']

#join new columns to dataset
X=pd.concat((pd.DataFrame(Recency),pd.DataFrame(Frequency),pd.DataFrame(Monetary)),axis=1)
X.columns=['R','F','M']

#scaling X
standardize=StandardScaler()
standardize.fit(X)
X=standardize.fit_transform(X)

X=pd.DataFrame(X)
X.columns=['R','F','M']

#Applying kmeans model for Clustering- we have 3 clusters
n_clusters=3
kmeans = KMeans(n_clusters=n_clusters).fit(X)

"""
If we do not know that how many cluster is appropriate for our model, we can use kmeans.inertia_ 
by plotting we can find the best number of clusters

wss=[]
for n in range(1,11):
 kmeans = KMeans(n_clusters=n).fit(X)
 wss.append(kmeans.inertia_)
 print(wss)

plt.plot(range(1,11),wss,c='r',marker='o',alpha=0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Within Sum of Squares Distance')
plt.show()
"""

#finding centroid of each cluster - find RMF for each cluster. Loyal customer should have high F and M but R should low
pd.DataFrame(kmeans.cluster_centers_,columns=X.columns)

#For better interpretating the result (After using scaling,interpretating the result is difficult. So the following line change X to original one )
Xorg=standardize.inverse_transform(X)
Xorg=pd.DataFrame(Xorg)
Xorg.columns=['R','F','M']
Xorg['clusters']=kmeans.labels_
print(Xorg.groupby('clusters').agg('mean'))

#get the unique labels
labels=kmeans.labels_
print(np.unique(labels))

#Find silhouette
average_silhouette=silhouette_score(X,labels)
print(average_silhouette)

#compute silhouette for each data
sample_silhouette_values=silhouette_samples(X,labels)
print(sample_silhouette_values)

#having no neagtive sample_silhouette_values is good
print(np.sum(sample_silhouette_values < 0))
