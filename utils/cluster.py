import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


def cluster_image(im, n_clusters, *, position_weight, mask=None):
    """
    Cluster image using k-means on combined position/color space

    Arguments:
        - im: image as a float ndarray
        - n_clusters: number of clusters
        - position_weight: relative weighting of pixel position vs. color

    Returns:
        - clustered_img: an ndarray where pixels are labeled by cluster index
        - colors: an ndarray of cluster colors by index
    """
    if mask is None:
        index_arr = np.indices((im.shape[0], im.shape[1])).reshape(
            (2, im.shape[0] * im.shape[1])
        )
    else:
        index_arr = np.array(np.where(mask))

    color_arr = im[index_arr[0], index_arr[1]]
    normalized_indices = index_arr.T / np.array([[im.shape[0], im.shape[1]]])

    X = np.hstack([normalized_indices * position_weight, color_arr])
    model = KMeans(n_clusters=n_clusters, n_init="auto")
    model.fit(X)
    colors = model.cluster_centers_[:, -3:]
    preds = model.predict(X)

    clustered_img = np.zeros((im.shape[0], im.shape[1]), dtype=int) - 1
    clustered_img[tuple(index_arr)] = preds

    return clustered_img, colors, preds


def get_closest(pairs):
    """
    Get closest pair of points. pairs is [(index, point), ...]

    TODO: there are more efficient ways to do this
    """
    pair_indices = (None, None)
    color_indices = (None, None)
    dist = float("inf")
    for i in range(len(pairs)):
        for j in range(i):
            new_dist = np.linalg.norm(pairs[i][1] - pairs[j][1])
            if new_dist < dist:
                pair_indices = (i, j)
                color_indices = (pairs[i][0], pairs[j][0])
                dist = new_dist

    return pair_indices, color_indices, dist


def combine_similar_clusters(clustered_im, colors, *, max_dist_to_combine):
    """
    Combine clusters which have similar colors until all clusters have colors
    that are `max_dist_to_combine` away from each other (using L2 norm)
    """
    clustered_im = clustered_im.copy()
    colors = colors.copy()

    cluster_sizes = np.array([(clustered_im == i).sum() for i in range(len(colors))])
    remaining_color_pairs = [(i, c) for i, c in enumerate(colors)]

    while True:
        (pair_i, pair_j), (color_i, color_j), dist = get_closest(remaining_color_pairs)
        if pair_i is None or pair_j is None or color_i is None or color_j is None:
            break

        if dist > max_dist_to_combine:
            break

        # make sure j is larger
        if pair_i > pair_j:
            pair_i, pair_j = pair_j, pair_i
            color_i, color_j = color_j, color_i

        del remaining_color_pairs[pair_j]

        if cluster_sizes[color_i] + cluster_sizes[color_j] > 0:
            colors[color_i] = (
                colors[color_i] * cluster_sizes[color_i]
                + colors[color_j] * cluster_sizes[color_j]
            ) / (cluster_sizes[color_i] + cluster_sizes[color_j])

        cluster_sizes[color_i] = cluster_sizes[color_i] + cluster_sizes[color_j]
        cluster_sizes[color_j] = 0

        clustered_im[clustered_im == color_j] = color_i

    new_clusters = np.zeros_like(clustered_im)
    new_colors = []
    next_color = 0
    for color_ind, c in remaining_color_pairs:
        if cluster_sizes[color_ind] == 0:
            continue
        new_clusters[clustered_im == color_ind] = next_color
        next_color += 1
        new_colors.append(c)

    return new_clusters, np.array(new_colors)


def split_region(mask, *, area_threshold):
    """
    Split a mask into regions which are smaller than the area threshold and
    larger than the area threshold
    """
    n, regions, stats, _ = cv.connectedComponentsWithStats(mask)
    large_region_mask = np.zeros_like(mask)
    small_region_mask = np.zeros_like(mask)

    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] >= area_threshold:
            large_region_mask[regions == i] = 1
        else:
            small_region_mask[regions == i] = 1

    return large_region_mask, small_region_mask


def terms_needed_for_err(fs, X, Y, *, mean_err_threshold, max_terms):
    """
    Find the number of terms needed in a fourier series for the average error
    to be `mean_err_threshold`. If this is larger than `max_terms`, it will
    return `max_terms` instead.
    """
    f = get_interpolator(X, Y)
    eps = 1e-6
    fx = f(np.linspace(eps, 1 - eps, 500))

    terms = 1
    while terms < max_terms:
        eq = fs_to_func(*fs, terms=terms)
        eqx = eq(np.linspace(eps, 1 - eps, 500))

        err = np.linalg.norm(fx - eqx, axis=1)

        if err.mean() < mean_err_threshold:
            break

        terms += 1

    return terms
