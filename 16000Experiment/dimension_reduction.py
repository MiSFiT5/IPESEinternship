# dimension_reduction.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn import metrics

def dimension_reduction(input_file, output_file, n_clusters_kmeans=4, n_clusters_kmedoids=5):
    df = pd.read_csv(input_file, header=None)
    scaler = MinMaxScaler()
    numerical_data_normalized = scaler.fit_transform(df.values)

    pca = PCA(n_components=3)
    numerical_data_pca = pca.fit_transform(numerical_data_normalized)
    numerical_data_pca_df = pd.DataFrame(numerical_data_pca)

    kmeans = KMeans(n_clusters=n_clusters_kmeans)
    kmeans.fit(numerical_data_pca)
    labels_kmeans = kmeans.labels_

    silhouette_score_kmeans = metrics.silhouette_score(numerical_data_pca, labels_kmeans)
    db_score_kmeans = metrics.davies_bouldin_score(numerical_data_pca, labels_kmeans)
    ch_score_kmeans = metrics.calinski_harabasz_score(numerical_data_pca, labels_kmeans)

    print("KMeans Clustering")
    print("Silhouette Score:", silhouette_score_kmeans)
    print("Davies-Bouldin Score:", db_score_kmeans)
    print("Calinski-Harabasz Score:", ch_score_kmeans)

    kmedoids = KMedoids(n_clusters=n_clusters_kmedoids, random_state=0)
    kmedoids.fit(numerical_data_pca)
    labels_kmedoids = kmedoids.labels_

    print("KMedoids Clustering")
    print("Cluster counts:", pd.Series(labels_kmedoids).value_counts())

    numerical_data_pca_df['Cluster_KMeans'] = labels_kmeans
    numerical_data_pca_df['Cluster_KMedoids'] = labels_kmedoids
    numerical_data_pca_df.to_csv(output_file, index=False)
    print(f"Dimension reduction and clustering completed, results saved to {output_file}")

if __name__ == "__main__":
    input_file = "encoded_data.csv"
    output_file = "dimensionReduced_data.csv"
    dimension_reduction(input_file, output_file)
