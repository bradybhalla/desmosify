import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import json

# from utils.clustering import cluster_image
from utils.fourier_series import fs_to_func, points_to_fs, fs_to_desmos


def show(image, title=""):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def laplacian_pyramid(im, levels):
    assert im.shape[0] % 2**levels == 0
    assert im.shape[1] % 2**levels == 0
    gaussian_pyramid = []
    for i in range(levels):
        gaussian_pyramid.append(im)
        im = cv.pyrDown(im)

    laplacian_pyramid = []
    for i in range(levels - 1):
        laplacian_pyramid.append(
            gaussian_pyramid[i] - cv.pyrUp(gaussian_pyramid[i + 1])
        )
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


def partial_collapse(laplacian_pyramid, high_detail_levels):
    width, height, _ = laplacian_pyramid[0].shape
    levels = len(laplacian_pyramid)

    low_detail = laplacian_pyramid[-1]
    for i in range(levels - high_detail_levels - 1):
        low_detail = cv.pyrUp(low_detail) + laplacian_pyramid[levels - i - 2]
    for i in range(high_detail_levels):
        low_detail = cv.pyrUp(low_detail)

    high_detail = np.zeros(
        (width // 2**high_detail_levels, height // 2**high_detail_levels, 3)
    )
    for i in range(high_detail_levels):
        high_detail = (
            cv.pyrUp(high_detail) + laplacian_pyramid[high_detail_levels - i - 1]
        )

    return low_detail, high_detail


def cluster_image(im, n_clusters, position_weight=0.0, mask=None):
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
    pairs is (index, color)

    TODO: there are more efficient functions if needed
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


def combine_similar_clusters(clustered_im, colors, max_dist_to_combine=20 / 255):
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


if __name__ == "__main__":
    # load image
    im = cv.imread("images/image.jpg")
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32) / 255
    height, width, _ = im.shape

    new_width = 512
    new_height = 512
    im = cv.resize(im, (new_width, new_height))
    # im = cv.bilateralFilter(im, 5, 5, 5)
    show(im)

    # find initial clustering
    clustered_im, colors, _ = cluster_image(im, 300, position_weight=1)
    print(f"done clustering ({len(colors)} clusters)", file=sys.stderr)
    show(colors[clustered_im])

    # combine clusters with similar color
    clustered_im, colors = combine_similar_clusters(clustered_im, colors)
    print(f"done combining clusters ({len(colors)} unique colors)", file=sys.stderr)
    show(colors[clustered_im])

    total_removed_area = 0

    all_fourier_series_eqns = []
    for i, c in enumerate(colors):
        r, g, b = (c * 255).astype(np.uint8)
        color_str = f"rgb({r},{g},{b})"

        mask = cv.flip((clustered_im == i).astype(np.uint8), 0)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for pixels in contours:

            area = cv.contourArea(pixels)

            # ignore regions that are too small
            if area < 10:
                total_removed_area += area
                continue

            if pixels.shape[0] <= 2:
                total_removed_area += area
                continue

            X = pixels[:, 0, 0]
            Y = pixels[:, 0, 1]

            fs = points_to_fs(X, Y)
            equation_object = {
                "latex": fs_to_desmos(*fs, max_terms=50),
                "color": color_str,
                "lines": True,
                "fill": True,
                "fillOpacity": 1,
                "lineWidth": 4,
            }
            all_fourier_series_eqns.append((area, json.dumps(equation_object)))


    print("done calculating equations", file=sys.stderr)
    print(f"missing area: {total_removed_area/512/512*100:.2f}%", file=sys.stderr)


    all_fourier_series_eqns.sort(key=lambda x: x[0], reverse=True)

    all_equations = [json.dumps({"latex": "N = \\left[0...30\\right]"})]
    all_equations += [i[1] for i in all_fourier_series_eqns]
    print(f"Calc.setExpressions([{','.join(all_equations)}])")
    print(f"num equations: {len(all_equations)}", file=sys.stderr)

    # show(ld_colors[clustered_im])

    # reconst = ld_colors[clustered_im]
    # show(reconst)

    # _, hd_colors, hd_clusters = cluster_image(im, 10, position_weight=0.1, mask=mask)
    #
    # new = np.zeros(im.shape)
    # new[np.where(mask)] = hd_colors[hd_clusters]
    # show(mask)
    # show(new)
    #
    # reconst[np.where(mask)] = hd_colors[hd_clusters]
    # show(reconst)
