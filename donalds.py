import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from statsmodels.discrete.discrete_model import Logit
from statsmodels.api import add_constant

data = pd.read_csv('mcdonalds.csv')

binary_columns = data.columns[:11]
data[binary_columns] = (data[binary_columns] == 'Yes').astype(int)

pca = PCA()
MD_x = data[binary_columns]
MD_pca = pca.fit_transform(MD_x)

plt.scatter(MD_pca[:, 0], MD_pca[:, 1], c='grey')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=1234)
kmeans_labels = kmeans.fit_predict(MD_x)

plt.scatter(MD_pca[:, 0], MD_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

linkage_matrix = linkage(MD_x.T, method='ward')
dendrogram(linkage_matrix, labels=binary_columns)
plt.show()

label_encoder = LabelEncoder()
data['Like.n'] = label_encoder.fit_transform(data['Like'])
data['Like.n'] = (data['Like.n'] - data['Like.n'].min()) / (data['Like.n'].max() - data['Like.n'].min())

X = data[binary_columns]
X = add_constant(X)
y = data['Like.n']
logit_model = Logit(y, X)
logit_results = logit_model.fit()

plt.plot(logit_results.pvalues, 'ro')
plt.axhline(y=0.05, color='b', linestyle='--')
plt.xticks(range(len(binary_columns) + 1), ['Intercept'] + list(binary_columns), rotation=90)
plt.xlabel('Features')
plt.ylabel('p-values')
plt.show()

linkage_matrix = linkage(MD_x.T, method='ward')
dendrogram(linkage_matrix, labels=binary_columns)
plt.show()

# Perform hierarchical clustering 
linkage_matrix = linkage(MD_x.T, method='ward')
hierarchical_labels = fcluster(linkage_matrix, 4, criterion='maxclust')

# Create a unique color for each label
unique_labels = np.unique(hierarchical_labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

# Plot the scatter plot with colors assigned to data points
for label, color in zip(unique_labels, colors):
    cluster_indices = np.where(hierarchical_labels == label)[0]
    plt.scatter(MD_pca[cluster_indices, 0], MD_pca[cluster_indices, 1], c=color, label=f'Cluster {label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
