import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.loadtxt("data_clustering.txt", delimiter=",")
print("Розмірність:", data.shape)

plt.figure(figsize=(6, 5))
plt.scatter(data[:, 0], data[:, 1], s=25, c='gray')
plt.title("Вихідні дані для кластеризації (LR_7_task_1)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
kmeans.fit(data)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.figure(figsize=(7, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=30)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X')
plt.title("Кластеризація методом K-середніх")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

print("Координати центрів кластерів:")
print(centers)
