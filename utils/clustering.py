import numpy as np
from sklearn.cluster import KMeans

# def cluster_image(im, n_clusters, position_weight=0.0):
#     """
#     Cluster image using k-means on combined position/color space
#
#     Arguments:
#         - im: image as a float ndarray
#         - n_clusters: number of clusters
#         - position_weight: relative weighting of pixel position vs. color
#
#     Returns:
#         - clustered_img: an ndarray where pixels are labeled by cluster index
#         - colors: an ndarray of cluster colors by index
#     """
#     index_arr = np.indices((im.shape[0],im.shape[1])).reshape((2,im.shape[0]*im.shape[1])).T 
#     color_arr = im[(*index_arr.T,)]
#     normalized_indices = index_arr / np.array([im.shape[0], im.shape[1]])
#
#     X = np.hstack([normalized_indices*position_weight, color_arr])
#     model = KMeans(n_clusters=n_clusters, n_init="auto")
#     model.fit(X)
#     colors = model.cluster_centers_[:,-3:]
#     preds = model.predict(X)
#
#     clustered_img = np.zeros((im.shape[0], im.shape[1]), dtype=int)
#     clustered_img[(*index_arr.T,)] = preds
#
#     return clustered_img, colors
