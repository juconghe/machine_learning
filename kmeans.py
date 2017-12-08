import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Data import Data


"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""


# ================= Helper Function for Plotting ====================
def plot_centroids(centroids, marker='o', scale=1):
    for c, centroid in enumerate(centroids):
        plt.plot([centroid[0]], [centroid[1]], color=cm(1. * c / K), marker=marker,
                 markerfacecolor='w', markersize=10*scale, zorder=scale)

# ===================================================================


class kmeans():
    def __init__(self, K):
        self.K = K      # Set the number of clusters

        np.random.seed(1234)
        self.centroids = np.random.rand(K, 2) # Initialize the position for those cluster centroids

        # Plot initial centroids with 'o'
        plot_centroids(self.centroids, marker='o')

    def fit(self, X, iterations=10):
        """
        :param X: Input data, shape: (N,2)
        :param iterations: Maximum number of iterations, Integer scalar
        :return: None
        """
        self.C = -np.ones(np.shape(X)[0],dtype=int) # Initializing which center does each sample belong to

        # WRITE the required CODE for learning HERE
        for itr in range(iterations):
            self.C = self.assign_clusters(X, self.C, self.centroids)
            self.centroids = self.update_centroids(X, self.C, self.centroids)

        return 0

    def update_centroids(self, X, C, centroids):
        """
        :param X: Input data, shape: (N,2)
        :param C: Assigned clusters for each sample, shape: (N,)
        :param centroids: Current centroid positions, shape: (K,2)
        :return: Update positions of centroids, shape: (K,2)

        Recompute centroids
        """
        # WRITE the required CODE HERE and return the computed values
        new_centroids = np.zeros((self.K,2))
        means = []
        for i in range(self.K):
            means.append(np.array([[0,0]]))

        for i in range(len(C)):
            x = X[i]
            centroid = int(C[i])
            # print(centroid)
            # print(means[centroid])
            means[centroid] = np.append(means[centroid],[x],axis=0)

        for mean_index in range(len(means)):
            m = means[mean_index][1:,]
            x_mean = np.mean(m[:,0])
            y_mean = np.mean(m[:,1])
            new_centroids[mean_index] = [x_mean, y_mean]

        return new_centroids

    def assign_clusters(self, X, C, centroids):
        """
        :param X: Input data, shape: (N,2)
        :param C: Assigned clusters for each sample, shape: (N,)
        :param centroids: Current centroid positions, shape: (K,2)
        :return: New assigned clusters for each sample, shape: (N,)

        Assign data points to clusters
        """
        # WRITE the required CODE HERE and return the computed values
        new_clusters = np.zeros(len(X))
        for x in range(len(X)):
            potential_centroid = 0
            dist = np.linalg.norm(X[x] - centroids[0])
            for i in range(len(centroids)):
                temp_dist = np.linalg.norm(X[x] - centroids[i])
                if temp_dist < dist:
                    potential_centroid = i
                    dist = temp_dist
            new_clusters[x] = potential_centroid

        return new_clusters

    def get_clusters(self):
        """
        *********** DO NOT EDIT *******
        :return: assigned clusters and centroid locaitons, shape: (N,), shape: (K,2)
        """
        return self.C, self.centroids

if __name__ == '__main__':
    cm = plt.get_cmap('gist_rainbow') # Color map for plotting

    # Get data
    data = Data()
    X = data.get_kmeans_data()

    # Compute clusters for different number of centroids
    for K in [2, 3, 5]:
        # K-means clustering
        model = kmeans(K)
        model.fit(X)
        C, centroids = model.get_clusters()

        # Plot final computed centroids with '*'
        plot_centroids(centroids, marker='*', scale=2)

        # Plot the sample points, with their color representing the cluster they belong to
        for i, x in enumerate(X):
            plt.plot([x[0]], [x[1]], 'o', color=cm(1. * C[i] / K), zorder=0)
        plt.axis('square')
        plt.axis([0, 1, 0, 1])
        plt.savefig('figures/Q3_' + str(K) + '.png')
        plt.close()
