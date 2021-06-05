# Import the necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import the data
air= pd.read_excel("EastWestAir.xlsx")
air.columns
air.describe()
air.head()
air.shape

# Normalizing or standardizing the data
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
df_norm= norm_func(air.iloc[:,1:])
df_norm.head()

# Normalizing or standardizing the data
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# Building hierarchial clustering dendrogram using different linkges
# Using complete linkage and euclidean method
z= linkage(df_norm, method='complete', metric='euclidean')
plt.figure(figsize=(15,5));plt.title('Hierarchical clustering dendrogram'); plt.xlabel("index"); plt.ylabel("distance")
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=8.,
)    

# Dendrogram of complete method is very clumsy, hence it is difficult to select number of clusters
# Using ward linkage and euclidean method
z= linkage(df_norm, method='ward', metric='euclidean')
plt.figure(figsize=(15,5));plt.title('Hierarchical clustering dendrogram');plt.xlabel('index'); plt.ylabel('distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=8.,
    )

# The dendrogram is clumsy at lower distances but at higher distances 4 distinctive clusters are formed at distance 1.0
# Considering the "complete" linkage cluster dendrogram, the distinctive clusters formed are 4.
from sklearn.cluster import AgglomerativeClustering
h_complete= AgglomerativeClustering(n_clusters=4, linkage="ward", affinity="euclidean").fit(df_norm)

cluster_labels= pd.Series(h_complete.labels_)
cluster_labels.head()

# creating a  new column and assigning the cluster label
air["clust"]=cluster_labels
air.head()
air.clust.value_counts()

# getting aggregate mean of each cluster
air.groupby(air.clust).mean()
air.iloc[:,1:].groupby(air.clust).mean()

# getting aggregate mean of each cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#KMEANS CLUSTERING
# screw plot or elbow curve
k = list(range(2,15))
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
plt.plot(k,TWSS,"ro-");plt.xlabel("numbar of clusters");plt.ylabel("total withiness squares");plt.xticks(k)

# Building KMeans model for 5 clusters
model=KMeans(n_clusters=5)
model.fit(df_norm)
model.labels_                # getting the labels of clusters assigned to each row

md= pd.Series(model.labels_) # converting numpy array into pandas series object
air["kclust"]= md                        # creating a  new column and assigning labels to it
air.head()
air.kclust.value_counts()

# Checking aggregate mean of each cluster
air.iloc[:,1:12].groupby(air.kclust).mean()
 
