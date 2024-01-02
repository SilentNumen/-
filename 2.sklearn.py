from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# 生成样本点
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels = make_blobs(n_samples=750, centers=centers,
                       cluster_std=0.4, random_state=0)

clustering = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap='prism')
plt.show()