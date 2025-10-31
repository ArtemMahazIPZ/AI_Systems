import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

data = np.loadtxt("data_clustering.txt", delimiter=",")

bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=100)
print(f"Оцінена ширина вікна (bandwidth): {bandwidth:.3f}")

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_
centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))

plt.figure(figsize=(7, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=35)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X')
plt.title(f"Кластеризація методом MeanShift (кластерів = {n_clusters})")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

print(f"Кількість знайдених кластерів: {n_clusters}")
print("Центри кластерів:")
print(centers)
