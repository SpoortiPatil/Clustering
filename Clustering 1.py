# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

# Import the data
crime= pd.read_csv("crime_data.csv")
crime.columns

# Changing the first column name for convinience
crime = crime.rename(columns={'Unnamed: 0':"States"})
crime.columns
crime.shape
crime.describe()

# summarize the number of unique values in each column
print(crime.nunique())

# Normalizing or standardizing the data
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm= norm_func(crime.iloc[:,1:])
df_norm.head()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# Building hierarchial clustering dendrogram using different linkges
# Using complete linkage and euclidean method
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# The dendrogram is clumsy at lower distances but at higher distances 4 distinctive clusters are formed at distance 1.0
# Considering the "complete" linkage cluster dendrogram, the distinctive clusters formed are 4.
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=4, linkage="complete", affinity="euclidean").fit(df_norm)

cluster_labels =pd.Series(h_complete.labels_) 
cluster_labels.head()

# creating a  new column and assigning the cluster labels
crime["clust"]=cluster_labels
crime.head()
crime.clust.value_counts()

# getting aggregate mean of each cluster
crime.groupby(crime.clust).mean()


# Importing necessary libraries
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#KMEANS CLUSTERING
# screw plot or elbow curve
k = list(range(2,15))            # Specifying the range of k for which the plot is to be created
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    WSS=[]                   # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1, df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))   
    
plt.plot(k, TWSS, "ro-");plt.xlabel("No_of_Clusters");plt.ylabel("Total_within_SS");plt.xticks(k)

model=KMeans(n_clusters=4)            # Building model for 4 clusters
model.fit(df_norm)
model.labels_ 

md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['kclust']=md # creating a  new column and assigning it to new column 
crime.head()

# Scatter plot of "Murder" and "Assault" for all clusters
plt.figure(figsize=(12,6))
sns.scatterplot(x=crime['Murder'], y = crime['Assault'],hue=crime.kclust)
crime.kclust.value_counts()  

# Checking aggregate mean of each cluster
crime.iloc[:,1:5].groupby(crime.kclust).mean()


