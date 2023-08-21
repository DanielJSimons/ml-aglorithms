import numpy as np
import matplotlib.pyplot as plt

class K_Means_Clustering:
    
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
        
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))
            
    def fit(self, X, max_iterations=100, plot_interval=5):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))
        
        for iteration in range(max_iterations):
            y = []
            
            for data_point in X:
                distances = K_Means_Clustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            
            y = np.array(y)
            cluster_indices = []
            
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
                
            cluster_centers = []
            
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
                    
            if np.max(self.centroids - np.array(cluster_centers)) < 0.001:
                break
            else:
                self.centroids = np.array(cluster_centers)
            
            if iteration % plot_interval == 0:
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='D', s=100)
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title(f'K-Means Clustering Iteration {iteration}')
                plt.savefig(f'iteration_{iteration}.png')
                plt.close()
            
            if iteration >= plot_interval and iteration % plot_interval == 0:
                input("Press Enter to continue...")
            
        return y

# Initializing random data values...
random_points = np.vstack([
    np.random.multivariate_normal(mean=[1, 2], cov=[[1, 0.5], [0.5, 1]], size=100),
    np.random.multivariate_normal(mean=[5, 6], cov=[[1, 0.5], [0.5, 1]], size=100),
    np.random.multivariate_normal(mean=[8, 2], cov=[[1, 0.5], [0.5, 1]], size=100)
])

kmeans = K_Means_Clustering(k=3)
labels = kmeans.fit(random_points, max_iterations=100, plot_interval=5)

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='D', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Example')
plt.show()
