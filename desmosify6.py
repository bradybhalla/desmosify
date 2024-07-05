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


def get_fs_for_mask(mask, min_outline_pixels = 2):
    assert min_outline_pixels >= 2

    # flip because y-axis is flipped in Desmos
    flipped = cv.flip(mask.astype(np.uint8), 0)
    contours, _ = cv.findContours(flipped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    fourier_series = []
    for pixels in contours:

        # ignore regions that are too small
        if pixels.shape[0] < min_outline_pixels:
            continue

        X = pixels[:, 0, 0]
        Y = pixels[:, 0, 1]

        fs = points_to_fs(X, Y)
        fourier_series.append(fs)

    return fourier_series


if __name__ == "__main__":
    # load image
    im = cv.imread("images/image.jpg")
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32) / 255
    height, width, _ = im.shape

    new_width = 512
    new_height = 512
    im = cv.resize(im, (new_width, new_height))

    pyramid = laplacian_pyramid(im, levels=9)
    low_detail, high_detail = partial_collapse(pyramid, high_detail_levels=0)


    mask = np.zeros(high_detail.shape[:2])
    mask[(high_detail**2).sum(axis=2) > 0.04] = 1


    clustered_im, ld_colors, _ = cluster_image(low_detail, 200, position_weight=2)
    h = ld_colors[clustered_im]
    h[mask == 1,:] = 1
    show(h)


    js_equation_objects = []
    for i, c in enumerate(ld_colors):
        fourier_series = get_fs_for_mask(clustered_im == i)
        for fs in fourier_series:
            r, g, b = (c * 255).astype(int)
            equation_object = {
                "latex": fs_to_desmos(*fs),
                "color": f"rgb({r},{g},{b})",
                "lines": True,
                "fill": True,
                "fillOpacity": 1,
            }
            js_equation_objects.append(json.dumps(equation_object))

    js_equation_objects.insert(0, json.dumps({"latex":"N=\\left[0...30\\right]"}))
    print(f"Calc.setExpressions([{','.join(js_equation_objects)}])")
    print(f"num equations: {len(js_equation_objects)}", file=sys.stderr)

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
