import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

dataset = pd.read_csv('dataset.csv')

kmeans = KMeans(n_clusters = 6)
clusters = kmeans.fit(dataset)
cluster_groups = kmeans.predict(dataset)
cluster_centroids = clusters.cluster_centers_
print(cluster_centroids)
