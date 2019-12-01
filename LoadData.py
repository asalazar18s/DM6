from pyclustering.cluster.cure import cure;
import numpy as np
import PIL as pl
import scipy as sp
import pandas as pd
from pyclustering.cluster import cluster_visualizer;
import matplotlib.pyplot as plt
from Normalization import normalize
from scipy.spatial import distance

filename = "Dataset(Clustering).csv"
col_names = list(range(1,102))
df = pd.read_csv(filename, sep=';', index_col=False, header=1)

print(df)

#df_array = normalize(df)

df_array = np.asarray(df)
#y = df_array.transpose()
#print(y)
print(df_array)
x = df_array.tolist()


cluster_instance = cure(x, 5, 4, 0.2)
cluster_instance.process()
clusters = cluster_instance.get_clusters()
representators = cluster_instance.get_representors()
means = cluster_instance.get_means()

avg_list = []  # list that will hold the avg distance of each cluster

for cluster in range(0, len(clusters)): # gets the list of each cluster 1->10
    cluster_mean = means[cluster]       # get the mean vector of the cluster
    curr_avg = 0                        # avg of current cluster
    for point in clusters[cluster]:     # loop through set of points allocated in the given cluster to get its avg dist
        clust_point = x[point]
        dist = distance.euclidean(cluster_mean,
                                  clust_point)
        curr_avg += dist
    total_clust_avg = curr_avg/len(clusters[cluster])
    avg_list.append(total_clust_avg)

final_avg = sum(avg_list)/len(avg_list)

