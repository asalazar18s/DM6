import pandas as pd
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pyclustering.cluster.cure import cure;
import numpy as np

palette = sns.color_palette("bright", 5)

# -------------------------------------------------------------------------------
# Read Data
filename = "Dataset(Clustering).csv"
col_names = list(range(1,102))
df = pd.read_csv(filename, sep=';', index_col=False, header=1)

# -------------------------------------------------------------------------------
# Transform DF to fir requirements
df_array = np.asarray(df)
x = df_array.tolist()

# -------------------------------------------------------------------------------
# Get Clusters
cluster_instance = cure(x, 5, 4, 0.2)
cluster_instance.process()
clusters = cluster_instance.get_clusters()
representors = cluster_instance.get_representors()
means = cluster_instance.get_means()

# -------------------------------------------------------------------------------
# Plot Clusters
labels = list(range(0,597))
cluster_ctr = 0
for cluster in clusters:
    for val in cluster:
        labels[val] = cluster_ctr
    cluster_ctr += 1


tsne = TSNE(n_components=2, random_state=0)
df_2d = tsne.fit_transform(df)

tsne_df = pd.DataFrame({'X':df_2d[:,0],
                        'Y':df_2d[:,1],
                        'cluster':labels})
print(tsne_df)

sns.scatterplot(x="X", y="Y",
                hue="cluster",
                palette=palette,
                legend='full',
                data=tsne_df)

plt.show()

